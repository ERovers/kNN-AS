import numpy as np
from sklearn.neighbors import NearestNeighbors

def KNN_sampling(Z, k, n):
    knn = NearestNeighbors(n_neighbors=k).fit(Z)
    distances, indices = knn.kneighbors(Z)
    scores = []

    size = Z.shape[-1]
    for i in range(indices.shape[0]):
        c = np.zeros((1,size))
        
        for ind in indices[i,1:]:
            c += Z[ind,:] - Z[i,:]

        scores.append(np.sqrt(np.sum(c**2)))
    return np.argsort(scores)[-n:]
