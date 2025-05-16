import pickle
import numpy as np
def heavy_tail(N_samp: int):
    radius = (-np.log(np.random.rand(N_samp)))**1.5
    signs = np.random.choice(a=[-1., 1.], size=(N_samp))
    return (radius * signs).reshape(-1,1)

if __name__ == '__main__':
    data = heavy_tail(1000)
    with open('data.pkl', 'wb') as f:
        pickle.dump({'labels': np.zeros(NN,dtype=np.int32), 'data': data}, f)
