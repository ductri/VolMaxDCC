import time

from our_model_training_utils import sub_main, main


if __name__ == "__main__":
    start = time.time()
    m = 10002
    lam = 0.0
    p = -1
    trial = 0
    list_hiddens = [2048, 10]
    dataset_name = 'stl10-simclr_pytorch2'

    sub_main(lam, False, p, trial, m, list_hiddens, dataset_name, epochs=100, lr=1e-3)

    # main(m, dataset_name)
    end = time.time()
    print(f'Total duration: {end-start:.2f}')

