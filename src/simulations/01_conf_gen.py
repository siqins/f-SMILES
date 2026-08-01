
from utils.conf_gen import *

if __name__ == '__main__':

    print("Conformer generation script")

    DATA_FOLDER = Path('data')
    ENUM_FOLDER = DATA_FOLDER / 'enum'
    ETKDG_FOLDER = DATA_FOLDER / 'conf-gen'
    XTB_FOLDER = DATA_FOLDER / 'xtb-calc'

    ENUM_FOLDER.mkdir(parents=True, exist_ok=True)
    ETKDG_FOLDER.mkdir(parents=True, exist_ok=True)
    XTB_FOLDER.mkdir(parents=True, exist_ok=True)


    input_files_name = 'data/enum/dataset.csv'
    output = 'data/conf-gen/dataset'
    N_CPU = 1

    # args = parser()
    # print(args)
    # if not args.output:
    #     output = Path(args.input).stem
    # else:
    #     output = args.output
    #
    # N_CPU = args.n_cpu
    INPUT_FILE = input_files_name
    CSV_FILE = f"{output}-UFFopt.csv"
    SDF_FILE_EMB = f"{output}-ETKDG.sdf"
    SDF_FILE_OPT = f"{output}-UFFopt.sdf"

    print("Conformer generation script")
    # Reading the CSV_FILE
    df = pd.read_csv(INPUT_FILE)
    print(df.head(), df.shape)

    print(f"{N_CPU} cpus allocated\n")
    with multiprocessing.Pool(N_CPU) as pool:
        t0 = time()
        print("Generating mol from smiles")
        moles = pool.map(mol_from_smiles, list(df.smi))
        t1 = time()

        print("Adding hydrogens to molecules")
        moles = pool.map(add_hs_to_mol, moles)
        t2 = time()

        print("Generating mol from smiles")
        moles = pool.map(gen3D, moles)
        t3 = time()

    embed = [flag for _, flag in moles]
    moles = [mol for mol, _ in moles]
    df['mol'] = moles
    df['mol-ETKDG'] = moles
    df['embed'] = embed

    with multiprocessing.Pool(N_CPU) as pool:
        t0 = time()
        print("Optimizing structures ...")
        moles = pool.map(uff_specialoptimize, list(df.mol))
        t1 = time()

    embed = [flag for _, flag in moles]
    moles = [mol for mol, _ in moles]

    df['mol'] = moles
    df['optim'] = embed

    # SDF file with optimized molecules
    t2 = time()
    PandasTools.WriteSDF(df, str(SDF_FILE_OPT), molColName="mol", properties=df.columns)
    t3 = time()

    # CSV file with optimized molecules
    t4 = time()
    df.drop('mol', axis=1).to_csv(CSV_FILE)
    t5 = time()

    # SDF file with embeded molecules
    t6 = time()
    PandasTools.WriteSDF(df, str(SDF_FILE_EMB), molColName="mol-ETKDG", properties=df.columns)
    t7 = time()

    # # Time
    print("\n\n")
    print(df.columns)

    print(f"uff_optimize: {t1 - t0:.0f}s")
    print(f"WriteSDF: {t3 - t2:.0f}s")
    print(f"WriteCSV: {t5 - t4:.0f}s")
    print(f"WriteSDF: {t7 - t6:.0f}s")