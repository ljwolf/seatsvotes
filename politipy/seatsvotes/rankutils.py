import numpy as np
from scipy.stats import rankdata, mode
from collections import OrderedDict 
from warnings import warn

def shares_at_rank(h,r=None, rk=None, bar_s=None):
    """
    Given a matrix of (n_sims, n_dists) voteshares, this provides the distributions of voteshares within
    each rank over all simulations.
    
    Arguments
    ----------
    h   :   (n_sims, n_dists) array
            matrix containing rows of a single simulated election's voteshares.
    r   :   (n_sims, n_dists) array
            matrix containing the rank each district occupies in the simulated election, row-wise.
            Computed on the fly if not provided. 
    rk  :   int or tuple
            rank (or tuple of ranks) to extract specifically.
    bar_s:  float
            a party seat share to use to target a specific level set of the normalized rank distribution.
            converted to a rank using the ceiling of bar_s*n_dists.
    return_dict: bool
                 a flag governing whether to return the full ordered dict relating ranks to voteshares, 
                 or to simply return the values array. Rarely, if there are ties too often in the simulations, 
                 the array will have fewer rows than ranks available in r. A warning will be raised in this case. 
    """
    h = np.asarray(h)
    if r is None:
        r = np.vstack([rankdata(1-hi, method='max') for hi in h])
    if bar_s is not None and rk is None:
        rk = np.ceil(bar_s*len(r[0]))
    tally = OrderedDict([(k,[]) for k in np.arange(1,len(r[0])+1,1)])
    for sim, rank in zip(h,r):
        for hi,ri in zip(sim,rank):
            tally[ri].append(hi)
    if rk is not None:
        if not isinstance(rk, tuple):
            rk = (rk,)
        tally = OrderedDict([(ri,tally[ri]) for ri in rk])
    return tally

class binreduce(object):
    """
    Collection of various reduction methods for the rank distribution corresponding to a given
    input data matrix.
    """

    @staticmethod
    def plot(h, r=None, n_bins=None,
             support=None, reduction=np.mean, reduce_kw=dict(), ax=None,
             fig_kw=dict(), plot_kw=dict()):
        import matplotlib.pyplot as plt
        reduction = binreduce.apply(h, r=r, n_bins=n_bins, support=support,
                                    reduction=reduction, **reduce_kw)
        support = np.linspace(h.min(), h.max(), num=n_bins) if support is None else support
        if ax is None:
            f = plt.figure(**fig_kw)
            ax = plt.gca()
        else:
            f = plt.gfc()
        ax.plot(support[:-1], reduction, **plot_kw)
        return f,ax

    @staticmethod
    def apply(h, r=None, n_bins=None, 
              support=None, reduction=np.mean, **reduce_args):
        """
        Given a matrix of simulated voteshares, this provides a resampled estimate
        the seats-votes curve given a specified grid and reduction. By default, this 
        will provide the mean rank for simulations that fall within a given gridcell
        """
        h = np.asarray(h)
        if n_bins is not None and support is not None:
            raise ValueError("Only bins or support may be provided, not both.")
        if r is None:
            r = np.vstack([rankdata(1-hi, method='max') for hi in h])
        if support is None:
            support = np.linspace(h.min(), h.max(), num=n_bins)
        bin_reduction = []
        for i,left_edge in enumerate(support):
            if i == len(support)-1:
                continue
            right_edge = support[i+1]
            mask = (left_edge <= h) & (h < right_edge)
            bin_reduction.append(reduction(r[mask], **reduce_args))
        return np.asarray(bin_reduction)
    
    @staticmethod
    def mean(h, r=None, n_bins=None, support=None):
        return binreduce.apply(h, r=r, n_bins=n_bins, support=support, reduction=np.mean)

    @staticmethod
    def median(h, r=None, n_bins=None, support=None):
        return binreduce.apply(h, r=r, n_bins=n_bins, support=support, reduction=np.median)

    @staticmethod
    def mode(h, r=None, n_bins=None, support=None):
        return binreduce.apply(h, r=r, n_bins=n_bins, support=support, reduction=mode)

    @staticmethod
    def count(h, r=None, n_bins=None, support=None):
        return binreduce.apply(h, r=r, n_bins=n_bins, support=support, reduction=len)

    @staticmethod
    def percentile(h, r=None, n_bins=None, support=None, q=[5,50,95]):
        return binreduce.apply(h, r=r, n_bins=n_bins, support=support, reduction=np.percentile,
                                      q=q)
    
    @staticmethod
    def in_band(h, r=None, lower=.45, upper=.55, 
                reduction=np.mean, **reduce_kw):
        if r is None:
            r = np.vstack([rankdata(1-hi, method='max') for hi in h])
        mask = (lower <= h) & (h <= upper)
        return reduction(r[mask], **reduce_kw)
