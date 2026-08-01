
from utils.xtb_run import *

def log(message: str) -> None:
    """
    Write a message to the log file.
    """
    global LOG_FILE
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")


if __name__ == '__main__':

    DATA_FOLDER = Path('data')
    ETKDG_FOLDER = DATA_FOLDER / 'conf-gen'
    XTB_FOLDER = DATA_FOLDER / 'xtb-calc'

    # for chrg in [0]:
    for chrg in [-1, 0, 1]:
        command = ["python", "utils/xtb_run.py",
                   "-f", f"{ETKDG_FOLDER / 'dataset-UFFopt.sdf'}",
                   "-opt",
                   "-hess",
                   "-acc", "vtight",
                   "-chrg", str(chrg),
                   "-o", f"{XTB_FOLDER / ('dataset_CHRG_' + str(chrg))}",
                   "-p", "10"]
        subprocess.run(command)
        print(" ".join(command))