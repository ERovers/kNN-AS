from sklearn.neighbors import NearestNeighbors
import numpy as np

def KNN_sampling(Z, k, n, sumv=True):
    """
    kNN-AS main algorithm to select new states either using
    the magnitude of vectorsum or average distances to their neighbours.
    Z (N_frames,d):
        Data from MD simulations.
    k (integer):
        Number of neighbours.
    n (integer):
        Number of states that are selected.
    """
    knn = NearestNeighbors(n_neighbors=k).fit(Z)
    distances, indices = knn.kneighbors(Z)
    scores = []
    if sumv:
        size = Z.shape[-1]
        for i in range(indices.shape[0]):
            c = np.zeros((1,size))

            for ind in indices[i,1:]:
                c += Z[ind,:] - Z[i,:] 

            scores.append(np.sqrt(np.sum(c**2)))
        return np.argsort(scores)[-n:]
    else:
        for i in range(indices.shape[0]):
            c = 0

            for ind in indices[i,1:]:
                c += np.sqrt(np.sum((Z[ind,:] - Z[i,:])**2))

            scores.append(c/k)
        return np.argsort(scores)[-n:]
