import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def empirical_qqplot(observed, target, num=20, ax=None,
                     fig_kw = dict(), scatter_kw = dict()):
    """
    Make a Q-Q plot that compares the observed distribution to the target distribution.
    Different from scipy.stats.probplot and statsmodels.api.graphics.qqplot in that it fits
    a comparison of observed and target where *both* are samples.

    Arguments
    ----------
    observed    :   arraylike
                    collection of observations from a distribution
    target      :   arraylike
                    sample from distribution being targeted
    num         :   int
                    number of evenly spaced percentiles to use. by default, moves in blocks of 5\%
    ax          :   AxesSubplot
                    if this plot is fitting inside of some other plot, pass this here.
    fig_kw      :   dict of keyword arguments
                    figure keywords to psas to the figure initialization
    scatter_kw  :   dict of keyword arguments
                    arguments to pass to the scatterplotting function drawing the qq plot.

    Returns
    ----------
    (Figure, AxesSubplot) or AxesSubplot containing the given Q-Q plot.
    """
    target = np.asarray(target)
    observed = np.asarray(observed)
    theoretical = np.percentile(target, np.linspace(1,99, num=num))
    empirical = np.percentile(observed, np.linspace(1,99, num=num))
    theoretical /= theoretical.max()
    empirical /= empirical.max()
    if ax is None:
        had_ax = False
        f,ax = plt.subplots(1,1, **fig_kw)
    else:
        had_ax = True
    ax.scatter(theoretical, empirical, **scatter_kw)
    ax.plot((0,1), (0,1), linestyle=':', color='k')
    ax.axis([0,1,0,1])
    ax.set_xlabel('Theoretical Percentiles')
    ax.set_ylabel('Empirical Percentiles')
    if had_ax:
        return ax
    else:
        return f,ax
