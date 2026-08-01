import os
import json
import time
import pickle
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.baselines.predictive_models.transformers import MolDatasetForMLM, TransformersForMLM

def save_tokens(tokens, filename='selfies_tokens.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)

def load_tokens(filename='selfies_tokens.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_epoch(mode, train_data, model, batch_size, optimizer, epoch, lr, lossfn):
    is_train = mode == 'train'

    if is_train:
        model.train()
    else:
        model.eval()

    loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    losses = []
    pbar = tqdm(enumerate(loader), total=len(loader))
    for it, (masked_seq, attention_mask, labels) in pbar:
        logits = model(masked_seq, attention_mask)
        loss = lossfn(logits.view(-1, logits.size(-1)), labels.view(-1))
        losses.append(loss.item())

        if is_train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
    return float(np.mean(losses))

def run_prediction(data, model, batch_size=64):
    model.eval()
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    pbar = tqdm(enumerate(loader), total=len(loader))
    preds = []
    for it, (masked_seq, attention_mask, labels) in pbar:
        logits = model.get_features(masked_seq, attention_mask)
        preds.append(logits.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy().squeeze()
    return preds

def save_checkpoint(model, ckpt_path):
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(raw_model.state_dict(), ckpt_path)

def parse(data_path, save_path, max_len: int, seq='smiles', model_save_name='model.pt', epochs: int=10):

    max_epochs: int = epochs
    batch_size: int = 384
    lr: float = 6e-4

    if os.path.join(save_path):
        os.makedirs(save_path, exist_ok=True)

    train_data_path = os.path.join(data_path, 'train_toks.pkl')
    valid_data_path = os.path.join(data_path, 'valid_toks.pkl')
    with open(train_data_path, 'rb') as f:
        train_toks = pickle.load(f)
    with open(valid_data_path, 'rb') as f:
        valid_toks = pickle.load(f)
    print('train_toks', len(train_toks))
    print('valid_toks', len(valid_toks))

    alphabet = load_tokens(os.path.join(data_path, f'{seq}_token_alphabet.json'))

    alphabet.insert(1, '[CLS]')
    alphabet.insert(2, '[MASK]')

    train_toks = [['[CLS]'] + toks for toks in train_toks]
    valid_toks = [['[CLS]'] + toks for toks in valid_toks]
    max_len = max_len + 1

    itos = {i:ch for i, ch in enumerate(alphabet)}
    stoi = {ch:i for i, ch in enumerate(alphabet)}
    train_toks = torch.tensor([stoi[tok] for toks in train_toks for tok in toks], dtype=torch.long).reshape(-1, max_len)
    valid_toks = torch.tensor([stoi[tok] for toks in valid_toks for tok in toks], dtype=torch.long).reshape(-1, max_len)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_dataset = MolDatasetForMLM(train_toks, vocab_size=len(alphabet), device=device)
    valid_dataset = MolDatasetForMLM(valid_toks, vocab_size=len(alphabet), device=device)
    train_dataset.set_special_tokens(mask_id=2, cls_id=1)
    valid_dataset.set_special_tokens(mask_id=2, cls_id=1)

    model = TransformersForMLM(vocab_len=len(alphabet), sent_len=max_len)
    model.to(device)

    set_seed(42)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    loss = torch.nn.CrossEntropyLoss(ignore_index=-100)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    best_loss = float('inf')
    train_losses, test_losses = [], []
    for epoch in range(max_epochs):
        train_loss = run_epoch('train', train_dataset, model, batch_size, optimizer, epoch, lr, loss)
        test_loss = run_epoch('valid', valid_dataset, model, batch_size, optimizer, epoch, lr, loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        good_model = test_loss < best_loss
        if good_model:
            best_loss = test_loss
            print(f'Saving at epoch {epoch + 1}')
            save_checkpoint(model, os.path.join(save_path, model_save_name))

    model_dict = torch.load(os.path.join(save_path, model_save_name))
    model.load_state_dict(model_dict)
    train_preds = run_prediction(train_dataset, model, batch_size)
    valid_preds = run_prediction(valid_dataset, model, batch_size)

    train_preds_df = pd.DataFrame(train_preds)
    valid_preds_df = pd.DataFrame(valid_preds)
    train_preds_df.to_csv(os.path.join(save_path, 'train_preds.csv'), index=False)
    valid_preds_df.to_csv(os.path.join(save_path, 'valid_preds.csv'), index=False)

    df = pd.DataFrame()
    df['epoch'] = range(max_epochs)
    df['train_loss'] = train_losses
    df['test_loss'] = test_losses
    df.to_csv(os.path.join(save_path, 'train_loss.csv'), index=False)

if __name__ == '__main__':

    data_path = r'../experiments/molgpt_exps'
    save_path = r'../experiments/sequence_mlm'

    max_len_dict = {
        'smiles' : 124,
        'smiles-pair-encoding': 51,
        'clear-smiles': 161,
        'deep-smiles': 143,
        'selfies': 126,
        'group-selfies': 107,
        'tsmiles': 42,
        'fsmiles': 15,
        'fsmiles-branched': 30
    }

    seqs = ['fsmiles', 'fsmiles-branched', 'clear-smiles', 'smiles', 'smiles-pair-encoding', 'deep-smiles', 'selfies', 'group-selfies', 'tsmiles']
    for seq in seqs:
        for idx in [0]:
            start_time = time.time()
            parse(data_path=os.path.join(data_path, seq),
                  save_path=os.path.join(save_path, seq),
                  max_len=max_len_dict[seq],
                  seq=seq,
                  model_save_name=f'model_50.pt',
                  epochs=50,
                  )
            end_time = time.time()
            print(seq, ' Total time:', end_time - start_time)
