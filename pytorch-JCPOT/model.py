import numpy as np
import warnings
from inspect import signature

from bregman import jcpot_barycenter
from utils import check_params, dist, cost_normalization, label_normalization


class BaseEstimator(object):
    """Base class for most objects in POT

    Code adapted from sklearn BaseEstimator class

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""

        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("POT estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimisation to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        # for key, value in iteritems(params):
        for key, value in params.items():
            split = key.split('__', 1)
            if len(split) > 1:
                # nested objects case
                name, sub_name = split
                if name not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (name, self))
                sub_object = valid_params[name]
                sub_object.set_params(**{sub_name: value})
            else:
                # simple objects case
                if key not in valid_params:
                    raise ValueError('Invalid parameter %s for estimator %s. '
                                     'Check the list of available parameters '
                                     'with `estimator.get_params().keys()`.' %
                                     (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

class BaseTransport(BaseEstimator):

    """Base class for OTDA objects

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).

    the fit method should:

    - estimate a cost matrix and store it in a `cost_` attribute
    - estimate a coupling matrix and store it in a `coupling_`
    attribute
    - estimate distributions from source and target data and store them in
    mu_s and mu_t attributes
    - store Xs and Xt in attributes to be used later on in transform and
    inverse_transform methods

    transform method should always get as input a Xs parameter
    inverse_transform method should always get as input a Xt parameter

    transform_labels method should always get as input a ys parameter
    inverse_transform_labels method should always get as input a yt parameter
    """

    def fit(self, Xs=None, ys=None, Xt=None, yt=None):
        """Build a coupling matrix from source and target sets of samples
        (Xs, ys) and (Xt, yt)

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The training class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self.
        """

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs, Xt=Xt):

            # pairwise distance
            self.cost_ = dist(Xs, Xt, metric=self.metric)
            self.cost_ = cost_normalization(self.cost_, self.norm)

            if (ys is not None) and (yt is not None):

                if self.limit_max != np.infty:
                    self.limit_max = self.limit_max * np.max(self.cost_)

                # assumes labeled source samples occupy the first rows
                # and labeled target samples occupy the first columns
                classes = [c for c in np.unique(ys) if c != -1]
                for c in classes:
                    idx_s = np.where((ys != c) & (ys != -1))
                    idx_t = np.where(yt == c)

                    # all the coefficients corresponding to a source sample
                    # and a target sample :
                    # with different labels get a infinite
                    for j in idx_t[0]:
                        self.cost_[idx_s[0], j] = self.limit_max

            # distribution estimation
            self.mu_s = self.distribution_estimation(Xs)
            self.mu_t = self.distribution_estimation(Xt)

            # store arrays of samples
            self.xs_ = Xs
            self.xt_ = Xt

        return self

    def fit_transform(self, Xs=None, ys=None, Xt=None, yt=None):
        """Build a coupling matrix from source and target sets of samples
        (Xs, ys) and (Xt, yt) and transports source samples Xs onto target
        ones Xt

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The training input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels for training samples
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        transp_Xs : array-like, shape (n_source_samples, n_features)
            The source samples samples.
        """

        return self.fit(Xs, ys, Xt, yt).transform(Xs, ys, Xt, yt)

    def transform(self, Xs=None, ys=None, Xt=None, yt=None, batch_size=128):
        """Transports source samples Xs onto target ones Xt

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The source input samples.
        ys : array-like, shape (n_source_samples,)
            The class labels for source samples
        Xt : array-like, shape (n_target_samples, n_features)
            The target input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels for target. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform

        Returns
        -------
        transp_Xs : array-like, shape (n_source_samples, n_features)
            The transport source samples.
        """

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs):

            if np.array_equal(self.xs_, Xs):

                # perform standard barycentric mapping
                transp = self.coupling_ / np.sum(self.coupling_, 1)[:, None]

                # set nans to 0
                transp[~ np.isfinite(transp)] = 0

                # compute transported samples
                transp_Xs = np.dot(transp, self.xt_)
            else:
                # perform out of sample mapping
                indices = np.arange(Xs.shape[0])
                batch_ind = [
                    indices[i:i + batch_size]
                    for i in range(0, len(indices), batch_size)]

                transp_Xs = []
                for bi in batch_ind:
                    # get the nearest neighbor in the source domain
                    D0 = dist(Xs[bi], self.xs_)
                    idx = np.argmin(D0, axis=1)

                    # transport the source samples
                    transp = self.coupling_ / np.sum(
                        self.coupling_, 1)[:, None]
                    transp[~ np.isfinite(transp)] = 0
                    transp_Xs_ = np.dot(transp, self.xt_)

                    # define the transported points
                    transp_Xs_ = transp_Xs_[idx, :] + Xs[bi] - self.xs_[idx, :]

                    transp_Xs.append(transp_Xs_)

                transp_Xs = np.concatenate(transp_Xs, axis=0)

            return transp_Xs

    def transform_labels(self, ys=None):
        """Propagate source labels ys to obtain estimated target labels as in [27]

        Parameters
        ----------
        ys : array-like, shape (n_source_samples,)
            The source class labels

        Returns
        -------
        transp_ys : array-like, shape (n_target_samples, nb_classes)
            Estimated soft target labels.

        References
        ----------

        .. [27] Ievgen Redko, Nicolas Courty, Rémi Flamary, Devis Tuia
           "Optimal transport for multi-source domain adaptation under target shift",
           International Conference on Artificial Intelligence and Statistics (AISTATS), 2019.

        """

        # check the necessary inputs parameters are here
        if check_params(ys=ys):

            ysTemp = label_normalization(np.copy(ys))
            classes = np.unique(ysTemp)
            n = len(classes)
            D1 = np.zeros((n, len(ysTemp)))

            # perform label propagation
            transp = self.coupling_ / np.sum(self.coupling_, 0, keepdims=True)

            # set nans to 0
            transp[~ np.isfinite(transp)] = 0

            for c in classes:
                D1[int(c), ysTemp == c] = 1

            # compute propagated labels
            transp_ys = np.dot(D1, transp)

            return transp_ys.T

    def inverse_transform(self, Xs=None, ys=None, Xt=None, yt=None,
                          batch_size=128):
        """Transports target samples Xt onto source samples Xs

        Parameters
        ----------
        Xs : array-like, shape (n_source_samples, n_features)
            The source input samples.
        ys : array-like, shape (n_source_samples,)
            The source class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The target input samples.
        yt : array-like, shape (n_target_samples,)
            The target class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform

        Returns
        -------
        transp_Xt : array-like, shape (n_source_samples, n_features)
            The transported target samples.
        """

        # check the necessary inputs parameters are here
        if check_params(Xt=Xt):

            if np.array_equal(self.xt_, Xt):

                # perform standard barycentric mapping
                transp_ = self.coupling_.T / np.sum(self.coupling_, 0)[:, None]

                # set nans to 0
                transp_[~ np.isfinite(transp_)] = 0

                # compute transported samples
                transp_Xt = np.dot(transp_, self.xs_)
            else:
                # perform out of sample mapping
                indices = np.arange(Xt.shape[0])
                batch_ind = [
                    indices[i:i + batch_size]
                    for i in range(0, len(indices), batch_size)]

                transp_Xt = []
                for bi in batch_ind:
                    D0 = dist(Xt[bi], self.xt_)
                    idx = np.argmin(D0, axis=1)

                    # transport the target samples
                    transp_ = self.coupling_.T / np.sum(
                        self.coupling_, 0)[:, None]
                    transp_[~ np.isfinite(transp_)] = 0
                    transp_Xt_ = np.dot(transp_, self.xs_)

                    # define the transported points
                    transp_Xt_ = transp_Xt_[idx, :] + Xt[bi] - self.xt_[idx, :]

                    transp_Xt.append(transp_Xt_)

                transp_Xt = np.concatenate(transp_Xt, axis=0)

            return transp_Xt

    def inverse_transform_labels(self, yt=None):
        """Propagate target labels yt to obtain estimated source labels ys

        Parameters
        ----------
        yt : array-like, shape (n_target_samples,)

        Returns
        -------
        transp_ys : array-like, shape (n_source_samples, nb_classes)
            Estimated soft source labels.
        """

        # check the necessary inputs parameters are here
        if check_params(yt=yt):

            ytTemp = label_normalization(np.copy(yt))
            classes = np.unique(ytTemp)
            n = len(classes)
            D1 = np.zeros((n, len(ytTemp)))

            # perform label propagation
            transp = self.coupling_ / np.sum(self.coupling_, 1)[:, None]

            # set nans to 0
            transp[~ np.isfinite(transp)] = 0

            for c in classes:
                D1[int(c), ytTemp == c] = 1

            # compute propagated samples
            transp_ys = np.dot(D1, transp.T)

            return transp_ys.T

class JCPOTTransport(BaseTransport):

    """Domain Adapatation OT method for multi-source target shift based on Wasserstein barycenter algorithm.

    Parameters
    ----------
    reg_e : float, optional (default=1)
        Entropic regularization parameter
    max_iter : int, float, optional (default=10)
        The minimum number of iteration before stopping the optimization
        algorithm if no it has not converged
    tol : float, optional (default=10e-9)
        Stop threshold on error (inner sinkhorn solver) (>0)
    verbose : bool, optional (default=False)
        Controls the verbosity of the optimization algorithm
    log : bool, optional (default=False)
        Controls the logs of the optimization algorithm
    metric : string, optional (default="sqeuclidean")
        The ground metric for the Wasserstein problem
    norm : string, optional (default=None)
        If given, normalize the ground metric to avoid numerical errors that
        can occur with large metric values.
    distribution_estimation : callable, optional (defaults to the uniform)
        The kind of distribution estimation to employ
    out_of_sample_map : string, optional (default="ferradans")
        The kind of out of sample mapping to apply to transport samples
        from a domain into another one. Currently the only possible option is
        "ferradans" which uses the method proposed in [6].

    Attributes
    ----------
    coupling_ : list of array-like objects, shape K x (n_source_samples, n_target_samples)
        A set of optimal couplings between each source domain and the target domain
    proportions_ : array-like, shape (n_classes,)
        Estimated class proportions in the target domain
    log_ : dictionary
        The dictionary of log, empty dic if parameter log is not True

    References
    ----------

    .. [1] Ievgen Redko, Nicolas Courty, Rémi Flamary, Devis Tuia
       "Optimal transport for multi-source domain adaptation under target shift",
       International Conference on Artificial Intelligence and Statistics (AISTATS),
       vol. 89, p.849-858, 2019.

    .. [6] Ferradans, S., Papadakis, N., Peyré, G., & Aujol, J. F. (2014).
            Regularized discrete optimal transport. SIAM Journal on Imaging
            Sciences, 7(3), 1853-1882.


    """

    def __init__(self, reg_e=.1, max_iter=10,
                 tol=10e-9, verbose=False, log=False,
                 metric="sqeuclidean",
                 out_of_sample_map='ferradans'):
        self.reg_e = reg_e
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.log = log
        self.metric = metric
        self.out_of_sample_map = out_of_sample_map

    def fit(self, Xs, ys=None, Xt=None, yt=None):

        """Building coupling matrices from a list of source and target sets of samples
        (Xs, ys) and (Xt, yt)

        Parameters
        ----------
        Xs : list of K array-like objects, shape K x (nk_source_samples, n_features)
            A list of the training input samples.
        ys : list of K array-like objects, shape K x (nk_source_samples,)
            A list of the class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label

        Returns
        -------
        self : object
            Returns self.
        """
        # check the necessary inputs parameters are here
        if check_params(Xs=Xs, Xt=Xt, ys=ys):

            self.xs_ = Xs
            self.xt_ = Xt

            returned_ = jcpot_barycenter(Xs=Xs, Ys=ys, Xt=Xt, reg=self.reg_e,
                                         metric=self.metric, distrinumItermax=self.max_iter, stopThr=self.tol,
                                         verbose=self.verbose, log=True)

            self.coupling_ = returned_[1]['gamma']

            # deal with the value of log
            if self.log:
                self.proportions_, self.log_ = returned_
            else:
                self.proportions_ = returned_
                self.log_ = dict()

        return self

    def transform(self, Xs=None, ys=None, Xt=None, yt=None, batch_size=128):
        """Transports source samples Xs onto target ones Xt

        Parameters
        ----------
        Xs : list of K array-like objects, shape K x (nk_source_samples, n_features)
            A list of the training input samples.
        ys : list of K array-like objects, shape K x (nk_source_samples,)
            A list of the class labels
        Xt : array-like, shape (n_target_samples, n_features)
            The training input samples.
        yt : array-like, shape (n_target_samples,)
            The class labels. If some target samples are unlabeled, fill the
            yt's elements with -1.

            Warning: Note that, due to this convention -1 cannot be used as a
            class label
        batch_size : int, optional (default=128)
            The batch size for out of sample inverse transform
        """


        # transformed source data
        # 就是根据运输计划K对源域数据进行转化
        transp_Xs = [] # [n_sources,n_sample_source,feature]

        # check the necessary inputs parameters are here
        if check_params(Xs=Xs):

            if all([np.allclose(x, y) for x, y in zip(self.xs_, Xs)]):

                # perform standard barycentric mapping for each source domain

                for coupling in self.coupling_:
                    transp = coupling / np.sum(coupling, 1)[:, None] # [n_sample_source,n_sample_target]
                    # set nans to 0
                    transp[~ np.isfinite(transp)] = 0

                    # compute transported samples
                    transp_Xs.append(np.dot(transp, self.xt_)) # np.dot(transp, self.xt_):[n_sample_source,feature]
            else:

                # perform out of sample mapping
                # 对任意一些数据进行转化
                indices = np.arange(Xs.shape[0]) # Xs:[n_samples,feature_length]

                batch_ind = [
                    indices[i:i + batch_size]
                    for i in range(0, len(indices), batch_size)]

                transp_Xs = []

                for bi in batch_ind:
                    transp_Xs_ = []

                    # get the nearest neighbor in the sources domains
                    xs = np.concatenate(self.xs_, axis=0) # [n_sources*n_sources_samples,feature_length]
                    # Xs[bi]:[batchsize,n_feature]   xs:[n_sources*n_sources_samples,n_feature]  dist(Xs[bi], xs):[batch_size,n_sources*n_sources_samples]
                    # idx:[batch_size,] the nearest sample's index of sources for every one in batchsize?
                    # idx 是与这堆数据比较接近的源域数据的下标，用它们的转化结果加到这堆数据上，就是这堆数据的结果
                    idx = np.argmin(dist(Xs[bi], xs), axis=1)

                    # transport the source samples
                    for coupling in self.coupling_:
                        transp = coupling / np.sum(
                            coupling, 1)[:, None]

                        transp[~ np.isfinite(transp)] = 0
                        transp_Xs_.append(np.dot(transp, self.xt_)) # np.dot(transp, self.xt_):[n_sample_source,feature]

                    transp_Xs_ = np.concatenate(transp_Xs_, axis=0)# [n_sources*n_sample_source,feature]

                    # define the transported points
                    # find the nearest source sample's index,then add the differ between the transformed source data and init source data to init Xs data as Xs's transformed result
                    transp_Xs_ = transp_Xs_[idx, :] + Xs[bi] - xs[idx, :] # [batchsize,feature]
                    transp_Xs.append(transp_Xs_)

                transp_Xs = np.concatenate(transp_Xs, axis=0) # [n_samples,feature]

            return transp_Xs

    def transform_labels(self, ys=None):
        """Propagate source labels ys to obtain target labels as in [27]

        Parameters
        ----------
        ys : list of K array-like objects, shape K x (nk_source_samples,)
            A list of the class labels

        Returns
        -------
        yt : array-like, shape (n_target_samples, nb_classes)
            Estimated soft target labels.
        """

        # check the necessary inputs parameters are here
        if check_params(ys=ys):
            yt = np.zeros((len(np.unique(np.concatenate(ys))), self.xt_.shape[0])) #[n_labels,n_target_sample]
            for i in range(len(ys)):
                # let labels start from a number
                ysTemp = label_normalization(np.copy(ys[i]))# ys[i]、ysTemp:[n_source_samples,]
                classes = np.unique(ysTemp)
                n = len(classes)
                ns = len(ysTemp)

                # perform label propagation
                transp = self.coupling_[i] / np.sum(self.coupling_[i], 1)[:, None]# coupling_[i]:[n_source_samples,n_target_samples]

                # set nans to 0
                transp[~ np.isfinite(transp)] = 0

                if self.log:
                    D1 = self.log_['D1'][i]
                else:
                    D1 = np.zeros((n, ns))# [n_labels,n_source_samples]

                    for c in classes:
                        D1[int(c), ysTemp == c] = 1

                # compute propagated labels
                # / len(ys)=/ k, means uniform sources transfering
                yt = yt + np.dot(D1, transp) / len(ys)# np.dot(D1, transp):[n_labels,n_target_samples] show the mass of every class for transfering to target samples

            return yt.T

    def inverse_transform_labels(self, yt=None):
        """Propagate source labels ys to obtain target labels

        Parameters
        ----------
        yt : array-like, shape (n_source_samples,)
            The target class labels

        Returns
        -------
        transp_ys : list of K array-like objects, shape K x (nk_source_samples, nb_classes)
            A list of estimated soft source labels
        """

        # check the necessary inputs parameters are here
        if check_params(yt=yt):
            transp_ys = []
            ytTemp = label_normalization(np.copy(yt))
            classes = np.unique(ytTemp)
            n = len(classes)
            D1 = np.zeros((n, len(ytTemp)))# [n_labels,n_target_samples] the true proportion of target data

            for c in classes:
                D1[int(c), ytTemp == c] = 1

            for i in range(len(self.xs_)):

                # perform label propagation
                transp = self.coupling_[i] / np.sum(self.coupling_[i], 1)[:, None]# transp、coupling_[i]:[n_source_samples,n_target_samples]

                # set nans to 0
                transp[~ np.isfinite(transp)] = 0

                # compute propagated labels
                # 每个源域样本转移到目标域固定某个类别的质量，每行之和为1
                # the mass for every source data to transfer to one class of target data，the sum of every row = 1
                transp_ys.append(np.dot(D1, transp.T).T)#np.dot(D1, transp.T).T:[n_source_samples,n_labels]

            return transp_ys