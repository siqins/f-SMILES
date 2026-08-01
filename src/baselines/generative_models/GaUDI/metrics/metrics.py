

from utils import fingerprints, average_agg_tanimoto


def internal_diversity(gen, n_jobs=1, device='cpu', fp_type='morgan',
                       gen_fps=None, p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    return 1 - (average_agg_tanimoto(gen_fps, gen_fps,
                                     agg='mean', device=device, p=p)).mean()


if __name__ == '__main__':

    import pandas as pd

    path = r'C:\Users\12233\OneDrive\项目一\PAS\diffusion\summary\hetro-test\result-mol.csv'

    data = pd.read_csv(path)['smi'].tolist()

    IntDiv = internal_diversity(data)
    print(IntDiv)

    IntDiv = internal_diversity(data, p=2)
    print(IntDiv)