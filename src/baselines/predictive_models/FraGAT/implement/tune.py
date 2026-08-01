from FraGAT_codes import *
import os
import json

def parse_data(smiles: list, values: list):
    return [{'SMILES': smi, 'Value': v} for smi, v in zip(smiles, values)]


def train(
        train_dataset: list=None, # a list in type of : {'SMILES': , 'Value': }
        valid_dataset: list=None,
        test_dataset: list=None,
        ExpName: str='',
        MainMetric: str='RMSE',
        DataPath: str='./data/ESQL_SMILESValue.txt',
        RootPath: str='./Experiments/',
        CUDA_VISIBLE_DEVICES: str='1',
        TaskNum: int=1,
        ClassNum: int=1,
        Augmentation: bool=False,
        Weight: bool=True,

        ValidRate: int=4000,
        PrintRate: int=20,
        Frag: bool=True,
        output_size: int=1,
        atom_feature_size: int=39,
        bond_feature_size: int=10,
        Feature: str='AttentiveFP',

        ValidBalance: bool=False,
        TestBalance: bool=False,
        MaxEpoch: int=800,

        Splitter: str='Random',

        UpdateRate: int=1,
        LowerThanMaxLimit: int=50,
        DecreasingLimit: int=30,

        FP_size: int= 150,
        atom_layers: int=3,
        mol_layers: int=2,
        BatchSize: int=200,
        drop_rate: float=0.2,
        lr: float=2.5,
        WeightDecay: int=5,

        SplitValidSeed: int= 38,
        SplitTestSeed: int= 8,
        TorchSeed: int= 2

):
    if test_dataset is not None:
        assert train_dataset is not None and valid_dataset is not None
    if valid_dataset is not None:
        assert train_dataset is not None

    ParamList = {
        'train_dataset': train_dataset,
        'valid_dataset': valid_dataset,
        'test_dataset': test_dataset,

        'ExpName': ExpName,
        'MainMetric': MainMetric,
        'DataPath': DataPath,
        'RootPath': RootPath,
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'TaskNum': TaskNum,
        'ClassNum': ClassNum,
        'Augmentation': Augmentation,
        'Weight': Weight,


        'ValidRate': ValidRate,
        'PrintRate': PrintRate,
        'Frag': Frag,
        'output_size': output_size,
        'atom_feature_size': atom_feature_size,
        'bond_feature_size': bond_feature_size,
        'Feature': Feature,

        'ValidBalance': ValidBalance,
        'TestBalance': TestBalance,
        'MaxEpoch': MaxEpoch,
        'SplitRate': [0.8,0.1],
        'Splitter': Splitter,

        'UpdateRate': UpdateRate,
        'LowerThanMaxLimit': LowerThanMaxLimit,
        'DecreasingLimit': DecreasingLimit,

        'FP_size': FP_size,
        'atom_layers': atom_layers,
        'mol_layers': mol_layers,
        'DNNLayers': [512],
        'BatchSize': BatchSize,
        'drop_rate': drop_rate,
        'lr': lr,
        'WeightDecay': WeightDecay,

        'SplitValidSeed': SplitValidSeed,
        'SplitTestSeed': SplitTestSeed,
        'TorchSeed': TorchSeed
    }

    # os.environ['CUDA_VISIBLE_DEVICES'] = ParamList['CUDA_VISIBLE_DEVICES']

    opt = Configs(ParamList)
    opt.add_args('SaveDir', opt.args['RootPath'] + opt.args['ExpName'] + '/')
    if not os.path.exists(opt.args['SaveDir']):
        os.makedirs(opt.args['SaveDir'])
    model_dir = opt.args['SaveDir'] + 'model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    ckpt, value = train_and_evaluate(opt)


def predict(
        opt,
        smiles_list: list,
        model_path: str = 'saved_models',
        batch_size: int = 200,
):
    checkpoint = t.load(model_path, map_location='cpu', weights_only=False)
    model = checkpoint['model']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    validset = [{'SMILES': smi, 'Value': 0} for smi in smiles_list]
    Validset = MolDatasetEval(validset, opt)
    validloader = t.utils.data.DataLoader(Validset, batch_size=1, shuffle=False, num_workers=0, \
                                          drop_last=False, worker_init_fn=np.random.seed(8), pin_memory=True)

    All_answer = []
    All_label = []
    for i in range(opt.args['TaskNum']):
        All_answer.append([])
        All_label.append([])
    # [tasknum, ]

    for ii, data in enumerate(validloader):
        # one molecule input, but batch is not 1. Different Frags of one molecule consist of a batch.
        [Input, Label] = data
        Input = [input.to(device) for input in Input]
        # Label = Label.to(device)  #     [wrongbatch, batch(mol), task, 1]
        # Label = Label.squeeze(-1)   #[wrongbatch, batch(mol), task]
        # Label = Label.squeeze(0)    #[batch(mol), task]
        #print(Label.size())
        # Label = Label.t()           #[task,batch(mol)]
        # for Label, different labels in a batch are actually the same, for they are exactly one molecule.
        # so the batch dim of Label is not exactly useful.

        output = model(Input)       #[batch, output_size]
        #print("Pred value before average is: ", output)
        output = output.mean(dim=0, keepdims=True)    #[1, output_size]

        #print("Target value is: ", Label)
        #print("Pred value is: ", output)
        #print(output.size())
        #output = model(AdjMat, FeatureMat)
        #print(output)

        for i in range(opt.args['TaskNum']):
            cur_task_output = output[:, i * opt.args['ClassNum'] : (i+1) * opt.args['ClassNum']]    # [1, ClassNum]
            # cur_task_label = Label[i][0]   # all of the batch are the same, so only picking [i][0] is enough.
            # if cur_task_label == -1:
            #     continue
            # else:
                # All_label[i].append(cur_task_label.item())
            for ii, data in enumerate(cur_task_output.tolist()):
                All_answer[i].append(data[0])

    # print('All_answer', All_answer)

    # len: num of tasks, contain x list of prediction for x tasks.
    return All_answer[0]


if __name__ == '__main__':

    smis = ['CC', 'CCOC', 'CCCCCOC', 'CCSOCC',
            'O=C1C2=C(C=CC=C2)C(/C1=C/C(C3)=CC4=C3C=CC5=C4C=CC6=C5C=CC=C6)=C(C#N)/C#N']
    smis = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smis]

    trainset = [{'SMILES': smi, 'Value': 0} for smi in smis]
    # the function parse_data is used to parse.

    # if the data path is used to send the data,
    # the data format is
    # the first column is SMILES, and the rest columns are properties,
    # not like default pandas format, the csv format dose not contain the names of rows.

    train(
        train_dataset=trainset,
        valid_dataset=trainset,

        MaxEpoch=5,
    )

    path = r'Experiments/ESQL/config.json'
    with open(path, 'r') as file:
        opt = json.load(file)

    opt= Configs(opt)
    predict(
        opt=opt,
        smiles_list=smis,
        model_path=r'Experiments/ESQL/model/model_optimizer_epoch4',
    )