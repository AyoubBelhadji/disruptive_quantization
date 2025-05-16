import pickle
import numpy as np
def ring(N_samp: int, ring_std: float):
    normsamps = np.random.randn(N_samp,2)
    ring_samps = normsamps / np.linalg.norm(normsamps, axis=1).reshape(-1,1)
    return ring_samps + np.random.randn(N_samp, 2)*ring_std
def asym_rings(N_samp_per: int, ring_std: float = 0.05):
    five_rings = [ring(N_samp_per, ring_std) for _ in range(5)]
    centers = [[-1.75,0],[0,0],[2.25,0],[-1.25,-0.5],[1.25,-0.5]]
    return np.concat([j*np.ones(N_samp_per,dtype=np.int32) for j in range(5)]), np.concat([r + np.array(c) for (r,c) in zip(five_rings, centers)])

if __name__ == '__main__':
    labels, rings = asym_rings(500)
    with open('data.pkl', 'wb') as f:
        pickle.dump({'data': rings, 'labels': labels}, f)
