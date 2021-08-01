import numpy as np
from scipy.spatial.distance import cdist

from backend import get_backend
def cost_normalization(C, norm=None):
    """ Apply normalization to the loss matrix

    Parameters
    ----------
    C : ndarray, shape (n1, n2)
        The cost matrix to normalize.
    norm : str
        Type of normalization from 'median', 'max', 'log', 'loglog'. Any
        other value do not normalize.

    Returns
    -------
    C : ndarray, shape (n1, n2)
        The input cost matrix normalized according to given norm.
    """

    if norm is None:
        pass
    elif norm == "median":
        C /= float(np.median(C))
    elif norm == "max":
        C /= float(np.max(C))
    elif norm == "log":
        C = np.log(1 + C)
    elif norm == "loglog":
        C = np.log1p(np.log1p(C))
    else:
        raise ValueError('Norm %s is not a valid option.\n'
                         'Valid options are:\n'
                         'median, max, log, loglog' % norm)
    return C
def label_normalization(y, start=0):
    """ Transform labels to start at a given value

    Parameters
    ----------
    y : array-like, shape (n, )
        The vector of labels to be normalized.
    start : int
        Desired value for the smallest label in y (default=0)

    Returns
    -------
    y : array-like, shape (n1, )
        The input vector of labels normalized according to given start value.
    """

    diff = np.min(np.unique(y)) - start
    if diff != 0:
        y -= diff
    return y

def check_params(**kwargs):
    """check_params: check whether some parameters are missing
    """

    missing_params = []
    check = True

    for param in kwargs:
        if kwargs[param] is None:
            missing_params.append(param)

    if len(missing_params) > 0:
        print("POT - Warning: following necessary parameters are missing")
        for p in missing_params:
            print("\n", p)

        check = False

    return check




def unif(n):
    """ return a uniform histogram of length n (simplex)

    Parameters
    ----------

    n : int
        number of bins in the histogram

    Returns
    -------
    h : np.array (n,)
        histogram of length n such that h_i=1/n for all i


    """
    return np.ones((n,)) / n

def euclidean_distances(X, Y, squared=False):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends.

    Parameters
    ----------
    X : {array-like}, shape (n_samples_1, n_features)
    Y : {array-like}, shape (n_samples_2, n_features)
    squared : boolean, optional
        Return squared Euclidean distances.

    Returns
    -------
    distances : {array}, shape (n_samples_1, n_samples_2)
    """

    nx = get_backend(X, Y)

    a2 = nx.einsum('ij,ij->i', X, X)
    b2 = nx.einsum('ij,ij->i', Y, Y)

    c = -2 * nx.dot(X, Y.T)
    c += a2[:, None]
    c += b2[None, :]

    c = nx.maximum(c, 0)

    if not squared:
        c = nx.sqrt(c)

    if X is Y:
        c = c * (1 - nx.eye(X.shape[0], type_as=c))

    return c

def dist(x1, x2=None, metric='sqeuclidean'):
    """Compute distance between samples in x1 and x2

    .. note:: This function is backend-compatible and will work on arrays
        from all compatible backends.

    Parameters
    ----------

    x1 : array-like, shape (n1,d)
        matrix with n1 samples of size d
    x2 : array-like, shape (n2,d), optional
        matrix with n2 samples of size d (if None then x2=x1)
    metric : str | callable, optional
        'sqeuclidean' or 'euclidean' on all backends. On numpy the function also
        accepts  from the scipy.spatial.distance.cdist function : 'braycurtis',
        'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
        'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis',
        'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
        'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.


    Returns
    -------

    M : array-like, shape (n1, n2)
        distance matrix computed with given metric

    """
    if x2 is None:
        x2 = x1
    if metric == "sqeuclidean":
        return euclidean_distances(x1, x2, squared=True)
    elif metric == "euclidean":
        return euclidean_distances(x1, x2, squared=False)
    else:
        if not get_backend(x1, x2).__name__ == 'numpy':
            raise NotImplementedError()
        else:
            return cdist(x1, x2, metric=metric)
