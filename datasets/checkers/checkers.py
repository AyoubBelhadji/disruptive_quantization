import numpy as np
import pickle
def checkers(N_samps: int, N_sq: int = 4):
    assert N_sq % 2 == 0
    rows = np.random.choice(np.arange(N_sq), N_samps)
    cols = np.random.choice(np.arange(N_sq // 2), N_samps)
    squares = np.column_stack([cols * 2 + np.mod(rows + 1, 2), rows])
    labels = rows*(N_sq//2) + cols
    return squares + np.random.rand(N_samps, 2), labels

if __name__ == '__main__':
    with open('data.pkl', 'wb') as f:
        data, labels = checkers(1000)
        pickle.dump({'data': data, 'labels': labels}, f)