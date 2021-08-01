import numpy as np

from utils import dist, unif


def projR(gamma, p):
    """return the KL projection on the row constrints """
    return np.multiply(gamma.T, p / np.maximum(np.sum(gamma, axis=1), 1e-10)).T


def projC(gamma, q):
    """return the KL projection on the column constrints """
    return np.multiply(gamma, q / np.maximum(np.sum(gamma, axis=0), 1e-10))

def jcpot_barycenter(Xs, Ys, Xt, reg, metric='sqeuclidean', numItermax=100,
                     stopThr=1e-6, verbose=False, log=False, **kwargs):
    r'''Joint OT and proportion estimation for multi-source target shift as proposed in [27]

    The function solves the following optimization problem:

    .. math::

        \mathbf{h} = arg\min_{\mathbf{h}}\quad \sum_{k=1}^{K} \lambda_k
                    W_{reg}((\mathbf{D}_2^{(k)} \mathbf{h})^T, \mathbf{a})

        s.t. \ \forall k, \mathbf{D}_1^{(k)} \gamma_k \mathbf{1}_n= \mathbf{h}

    where :

    - :math:`\lambda_k` is the weight of k-th source domain
    - :math:`W_{reg}(\cdot,\cdot)` is the entropic regularized Wasserstein distance (see ot.bregman.sinkhorn)
    - :math:`\mathbf{D}_2^{(k)}` is a matrix of weights related to k-th source domain defined as in [p. 5, 27], its expected shape is `(n_k, C)` where `n_k` is the number of elements in the k-th source domain and `C` is the number of classes
    - :math:`\mathbf{h}` is a vector of estimated proportions in the target domain of size C
    - :math:`\mathbf{a}` is a uniform vector of weights in the target domain of size `n`
    - :math:`\mathbf{D}_1^{(k)}` is a matrix of class assignments defined as in [p. 5, 27], its expected shape is `(n_k, C)`

    The problem consist in solving a Wasserstein barycenter problem to estimate the proportions :math:`\mathbf{h}` in the target domain.

    The algorithm used for solving the problem is the Iterative Bregman projections algorithm
    with two sets of marginal constraints related to the unknown vector :math:`\mathbf{h}` and uniform target distribution.

    Parameters
    ----------
    Xs : list of K np.ndarray(nsk,d)
        features of all source domains' samples
    Ys : list of K np.ndarray(nsk,)
        labels of all source domains' samples
    Xt : np.ndarray (nt,d)
        samples in the target domain
    reg : float
        Regularization term > 0
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on relative change in the barycenter (>0)
    log : bool, optional
        record log if True
    verbose : bool, optional (default=False)
        Controls the verbosity of the optimization algorithm

    Returns
    -------
    h : (C,) ndarray
        proportion estimation in the target domain
    log : dict
        log dictionary return only if log==True in parameters


    References
    ----------

    .. [27] Ievgen Redko, Nicolas Courty, Rémi Flamary, Devis Tuia
       "Optimal transport for multi-source domain adaptation under target shift",
       International Conference on Artificial Intelligence and Statistics (AISTATS), 2019.

    '''
    nbclasses = len(np.unique(Ys[0]))
    nbdomains = len(Xs)

    # log dictionary
    if log:
        log = {'niter': 0, 'err': [], 'M': [], 'D1': [], 'D2': [], 'gamma': []}

    K = []
    M = []
    D1 = []
    D2 = []

    # For each source domain, build cost matrices M, Gibbs kernels K and corresponding matrices D_1 and D_2
    for d in range(nbdomains):
        dom = {}
        nsk = Xs[d].shape[0]  # get number of elements for this domain
        dom['nbelem'] = nsk
        classes = np.unique(Ys[d])  # get number of classes for this domain

        # format classes to start from 0 for convenience
        if np.min(classes) != 0:
            Ys[d] = Ys[d] - np.min(classes)
            classes = np.unique(Ys[d])

        # build the corresponding D_1 and D_2 matrices
        Dtmp1 = np.zeros((nbclasses, nsk))
        Dtmp2 = np.zeros((nbclasses, nsk))

        for c in classes:
            nbelemperclass = np.sum(Ys[d] == c)
            if nbelemperclass != 0:
                Dtmp1[int(c), Ys[d] == c] = 1.
                Dtmp2[int(c), Ys[d] == c] = 1. / (nbelemperclass)
        D1.append(Dtmp1)
        D2.append(Dtmp2)

        # build the cost matrix and the Gibbs kernel
        Mtmp = dist(Xs[d], Xt, metric=metric)
        M.append(Mtmp)

        Ktmp = np.empty(Mtmp.shape, dtype=Mtmp.dtype)
        np.divide(Mtmp, -reg, out=Ktmp)
        np.exp(Ktmp, out=Ktmp)
        K.append(Ktmp)

    # uniform target distribution
    a = unif(np.shape(Xt)[0])

    cpt = 0  # iterations count
    err = 1
    old_bary = np.ones((nbclasses))

    while (err > stopThr and cpt < numItermax):

        bary = np.zeros((nbclasses))

        # update coupling matrices for marginal constraints w.r.t. uniform target distribution
        for d in range(nbdomains):
            K[d] = projC(K[d], a)
            other = np.sum(K[d], axis=1)
            bary = bary + np.log(np.dot(D1[d], other)) / nbdomains

        bary = np.exp(bary)

        # update coupling matrices for marginal constraints w.r.t. unknown proportions based on [Prop 4., 27]
        for d in range(nbdomains):
            new = np.dot(D2[d].T, bary)
            K[d] = projR(K[d], new)

        err = np.linalg.norm(bary - old_bary)
        cpt = cpt + 1
        old_bary = bary

        if log:
            log['err'].append(err)

        if verbose:
            if cpt % 200 == 0:
                print('{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))

    bary = bary / np.sum(bary)


    # print(np.array(K).shape)

    if log:
        log['niter'] = cpt
        log['M'] = M
        log['D1'] = D1
        log['D2'] = D2
        log['gamma'] = K # [n_sources,n_source_simples,n_target_simples]
        return bary, log
    else:
        return bary