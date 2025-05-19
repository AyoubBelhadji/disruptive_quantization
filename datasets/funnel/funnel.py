import numpy as np
import pickle

def funnel(N_samps: int):
    v_samps = 3*np.random.randn(N_samps)
    x1_samps = np.exp(0.5*v_samps) * np.random.randn(N_samps)
    return np.column_stack([v_samps, x1_samps]), np.zeros(N_samps, dtype=np.int32)

if __name__ == '__main__':
    samples, labels = funnel(1000)
    with open('data.pkl', 'wb') as f:
        pickle.dump({'data': samples, 'labels': labels}, f)