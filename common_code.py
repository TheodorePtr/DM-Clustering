from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

def test_hopkins(X: pd.DataFrame, sample_size: int = None) -> float:
    """
    Calculates the Hopkins statistic for the input DataFrame:
        If the value is between (0.01, ...,0.3), the data is regularly spaced.

        If the value is around 0.5, it is random.

        If the value is between (0.7, ..., 0.99), it has a high tendency to cluster.
        
    Parameters:
        X (pd.DataFrame): The input DataFrame of shape (n_samples, n_features).
        sample_size (int): The size of the random sample to use. Defaults to 10% of the input size.

    Returns:
        The Hopkins statistic for the input DataFrame.
    """

    if sample_size is None:
        sample_size = int(0.25 * len(X))

    nbrs = NearestNeighbors(n_neighbors=1,
                            algorithm='brute',
                            metric='euclidean')
    nbrs.fit(X.values)
    
    rand_X = np.random.choice(len(X), sample_size, replace=False)

    ujd = []
    wjd = []
    for j in rand_X:
        u_dist, _ = nbrs.kneighbors(np.random.uniform(np.amin(X.values, axis=0),
                                                      np.amax(X.values, axis=0),
                                                      size=X.shape[1]).reshape(1, -1),
                                    2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.values[j].reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    h = 1 if (sum(ujd) + sum(wjd)) == 0 else sum(ujd) / (sum(ujd) + sum(wjd))
    if np.isnan(h):
        print(ujd, wjd)
        h = 0

    return h