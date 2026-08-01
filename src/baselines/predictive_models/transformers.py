# -*- coding: utf-8 -*-
# @Author  : Song Siq
# @Time    : 2024/10/26 18:46
# @Function:


import torch
from torch import nn
from torch.utils.data import Dataset

class MolDataset(Dataset):

    def __init__(self, tokens, labels=None, device='cpu'):
        self.tokens = tokens
        self.max_len = len(tokens[0])
        self.labels = labels
        self.device = device

    def __getitem__(self, idx):
        seq = self.tokens[idx].long()
        mask = (self.tokens[idx] != 0).detach().clone().long() # 不是pad的位置
        key_padding_mask = (self.tokens[idx] == 0).detach().clone()
        label = 0 if self.labels is None else torch.tensor(self.labels[idx]).float()

        seq = seq.to(self.device)
        mask = mask.to(self.device)
        key_padding_mask = key_padding_mask.to(self.device)
        label = label.to(self.device)

        return [seq, mask, key_padding_mask], label

    def __len__(self):
        return len(self.tokens)


class Transformers(nn.Module):

    def __init__(
            self,
            vocab_len: int,
            sent_len: int,
            embed_dim=256,
            dim_feedforward=256,
            hidden_dim=256,
            out_dim=1,
            n_heads=4,
            dropout=0.1,
            activation='relu',
            n_layers=4
    ):
        super().__init__()
        '''
        vocab_len：length of vocab
        sent_len：the supported maximum length of text
        '''
        super().__init__()
        self.sent_embed = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embed_dim)
        self.seg_embed = nn.Embedding(2, embedding_dim=embed_dim)
        self.position_embed = nn.Parameter(torch.randn(sent_len, embed_dim)/torch.tensor(10))

        encode_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        norm = nn.LayerNorm(normalized_shape=embed_dim, elementwise_affine=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encode_layer,
            num_layers=n_layers,
            norm=norm,
            enable_nested_tensor=False
        )

        self.feat_lin = nn.Linear(embed_dim, hidden_dim)

        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, inputs):

        seq, mask, key_padding_mask = inputs
        embed = self.sent_embed(seq) + self.seg_embed(mask) + self.position_embed

        x = self.encoder(src=embed, src_key_padding_mask=key_padding_mask)[:, 0]

        x = self.feat_lin(x)

        return self.pred_head(x)


class MolDatasetForMLM(Dataset):
    def __init__(self, tokens, vocab_size, device='cpu', mask_prob=0.15):
        self.tokens = tokens
        self.vocab_size = vocab_size
        self.device = device
        self.mask_prob = mask_prob

        self.mask_token_id = None
        self.cls_token_id = None
        self.pad_token_id = 0

    def set_special_tokens(self, mask_id, cls_id):
        self.mask_token_id = mask_id
        self.cls_token_id = cls_id

    def mask_tokens(self, tokens):
        """token masking"""
        labels = tokens.clone()
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        special_tokens_mask = (tokens == self.cls_token_id) | (tokens == self.pad_token_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        mask_indices = masked_indices & (torch.rand(len(tokens)) < 0.8)
        tokens[mask_indices] = self.mask_token_id
        random_indices = masked_indices & (torch.rand(len(tokens)) < 0.5) & ~mask_indices
        random_tokens = torch.randint(self.vocab_size, tokens.shape, dtype=torch.long)
        tokens[random_indices] = random_tokens[random_indices]
        labels[~masked_indices] = -100

        return tokens, labels

    def __getitem__(self, idx):
        seq = self.tokens[idx].clone()
        masked_seq, labels = self.mask_tokens(seq)
        attention_mask = (seq != self.pad_token_id).long()
        masked_seq = masked_seq.to(self.device)
        labels = labels.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return masked_seq, attention_mask, labels

    def __len__(self):
        return len(self.tokens)


class TransformersForMLM(nn.Module):
    def __init__(
            self,
            vocab_len: int,
            sent_len: int,
            embed_dim=256,
            dim_feedforward=256,
            hidden_dim=256,
            n_heads=4,
            dropout=0.1,
            activation='relu',
            n_layers=4
    ):
        super().__init__()

        self.sent_embed = nn.Embedding(num_embeddings=vocab_len, embedding_dim=embed_dim)
        self.seg_embed = nn.Embedding(2, embedding_dim=embed_dim)
        self.position_embed = nn.Parameter(torch.randn(sent_len, embed_dim) / torch.tensor(10))

        encode_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        norm = nn.LayerNorm(normalized_shape=embed_dim, elementwise_affine=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encode_layer,
            num_layers=n_layers,
            norm=norm,
            enable_nested_tensor=False
        )

        self.mlm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_len)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len], 1 for real tokens, 0 for padding
        """
        embed = self.sent_embed(input_ids)
        if attention_mask is not None:
            seg_embed = self.seg_embed(attention_mask)
        if attention_mask is not None:
            embed = embed + self.position_embed
        else:
            embed = embed + seg_embed + self.position_embed

        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
            x = self.encoder(src=embed, src_key_padding_mask=key_padding_mask)
        else:
            x = self.encoder(src=embed)
        logits = self.mlm_head(x)

        return logits

    def get_features(self, input_ids, attention_mask=None):
        # embed = self.sent_embed(input_ids) + self.position_embed

        embed = self.sent_embed(input_ids)
        embed = embed + self.position_embed

        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
            feats = self.encoder(src=embed, src_key_padding_mask=key_padding_mask)
        else:
            feats = self.encoder(src=embed)

        return feats[:, 0, :]
