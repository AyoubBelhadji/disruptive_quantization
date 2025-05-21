import pickle
import numpy as np
def heavy_tail(N_samp: int):
    radius = (-np.log(np.random.rand(N_samp)))**1.5
    angle = np.random.rand(N_samp)*2*np.pi
    return np.column_stack([np.cos(angle)*radius, np.sin(angle)*radius])

if __name__ == '__main__':
    NN = 1000
    data = heavy_tail(NN)
    with open('data.pkl', 'wb') as f:
        pickle.dump({'labels': np.zeros(NN,dtype=np.int32), 'data': data}, f)
