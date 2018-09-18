from sklearn import mixture as mx
from sklearn.utils import check_random_state
import numpy as np
import warnings as warn

def optimize_degree(data, max_degree = 10, ic='bic', ic_kw=None,
                    mix_kw=None, fit_kw=None):
    """
    Pick the best degree gaussian mixture based on the given information criteria.

    Arguments
    ---------
    data        :   array-like
                    data that will be passed to scikit-learn
    max_degree  :   int
                    size of the search to conduct. Typically, the number of mixtures should be less than twice the number of competing parties.
    ic          :   str
                    a string denoting the information criteria to use. Must be available in the `GaussianMixture`. Defaults to `bic`, the Bayesian Information Criterion, but can also be `aic` or `score`.
    ic_kw       :   dict of keyword arguments
                    keyword arguments passed down to the information criterion call. See also the relevant `GaussianMixture.bic` call
    mix_kw      :   dict of keyword arguments
                    keyword arguments passed down to the actual `GaussianMixture` call. See also `sklearn.mixture.GaussianMixture`.
    fit_kw      :   dict of keyword arguments
                    Keyword arguments passed down to the actual `GaussianMixture.fit()` command. See also
                    `sklearn.mixture.GaussianMixture.fit`
    """
    if mix_kw is None:
        mix_kw = dict()
    if fit_kw is None:
        fit_kw = dict()
    if ic_kw is None:
        ic_kw = dict()
    if 'n_components' in mix_kw:
        warn('This function finds the best number of components. '
             'Do not pass in `n_components`, they will be ignored.')
        del mix_kw['n_components']
    if ic not in ('aic', 'bic'):
        raise AttributeError('GaussianMixture has no '
                              'information criterion {}'.format(ic))
    models = (mx.GaussianMixture(n_components = i, **mix_kw).fit(data, **fit_kw)
                for i in range(1,max_degree))
    ics = [getattr(model, ic)(data, **ic_kw) for model in models]
    return np.argmin(ics)+1, ics

def step_out(data, target_degree=1, step=1, ic='bic', ic_kw=dict(),
             mix_kw=dict(), fit_kw=dict()):
    """
    Estimates the nearest two mixture with `step` more and `step` fewer mixture components.

    Arguments
    ----------
    data            :   array-like
                        data to fit the mixture model
    target_degree   :   int
                        starting degree from which to step out
    step            :   int
                        number of components to step out around target_degree
    ic              :   str
                        name of information criterion to use in the model check. See `sklearn.mixture.GaussianMixture` for more.
    ic_kw       :   dict of keyword arguments
                    keyword arguments passed down to the information criterion call. See also the relevant `GaussianMixture.bic` call
    mix_kw      :   dict of keyword arguments
                    keyword arguments passed down to the actual `GaussianMixture` call. See also `sklearn.mixture.GaussianMixture`.
    fit_kw      :   dict of keyword arguments
                    Keyword arguments passed down to the actual `GaussianMixture.fit()` command. See also
                    `sklearn.mixture.GaussianMixture.fit`
    """
    models = (mx.GaussianMixture(n_components = i, **mix_kw)
                .fit(data, **fit_kw)
              for i in [target_degree - step, target_degree, target_degree + step])
    models, ics = zip(*[(model, getattr(model, ic)(data, **ic_kw)) for model in models])
    return models, ics

def _fit_mixture(datum, n_components = None, **kw):
    N,P = datum.shape
    if n_components is None:
        n_components, info = optimize_degree(datum, max_degree = P*2,
                                               **kw)
    mixture = mx.GaussianMixture(n_components=n_components, **kw.get('mix_kw', dict()))
    fit = mixture.fit(datum, **kw.get('fit_kw', dict()))
    return fit

def fit_mixtures(*data, n_components = None, **kw):
    M = len(data)
    if M == 1:
        return _fit_mixture(data[0], n_components=n_components, **kw)
    else:
        return [_fit_mixture(datum, n_components=n_components, **kw) for datum in data]

def fixed_sample(mixture_model, n_samples):
    """
    This reimplements a sampling technique from scikit gaussian mixture models, in response to scikit-learn#7701
    """
    if n_samples < 1:
        raise ValueError(
            "Invalid value for 'n_samples': %d . The sampling requires at "
            "least one sample." % (mixture_model.n_components))

    _, n_features = mixture_model.means_.shape
    rng = check_random_state(mixture_model.random_state)
    n_samples_comp = rng.multinomial(n_samples, mixture_model.weights_).astype(int)

    if mixture_model.covariance_type == 'full':
        X = np.vstack([
            rng.multivariate_normal(mean, covariance, int(sample))
            for (mean, covariance, sample) in zip(
                mixture_model.means_, mixture_model.covariances_, n_samples_comp)])
    elif mixture_model.covariance_type == "tied":
        X = np.vstack([
            rng.multivariate_normal(mean, mixture_model.covariances_, int(sample))
            for (mean, sample) in zip(
                mixture_model.means_, n_samples_comp)])
    else:
        X = np.vstack([
            mean + rng.randn(sample, n_features) * np.sqrt(covariance)
            for (mean, covariance, sample) in zip(
                mixture_model.means_, mixture_model.covariances_, n_samples_comp)])

    y = np.concatenate([j * np.ones(sample, dtype=int)
                       for j, sample in enumerate(n_samples_comp)])

    return (X, y)