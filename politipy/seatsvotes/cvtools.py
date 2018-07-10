import numpy as np
import statsmodels as __sm
import pandas as pd

sm = __sm.api

def leverage(results):
    """
    Compute the leverage matrix from a WLS model:

    H = W^.5 X(X'WX)^-1X'W^.5

    where W^.5 is the scalar square root.
    """
    if isinstance(results.model, sm.WLS):
        W = np.diag(results.model.weights)
    elif isinstance(results.model, sm.OLS):
        W = np.eye(results.model.exog.shape[0])
    X = results.model.exog
    emp_cov = results.cov_params(scale=1)
    unW_H = X.dot(emp_cov).dot(X.T)
    return (W**.5).dot(unW_H).dot(W**.5)

def jackknife(candidate, full=False, **kw):
    """
    Estimate each leave-one-out model given the input data. 

    Arguments
    ---------
    candidate: model
               statsmodels regression model or a pysal regression model.
    full     : bool
               whether to return all reestimated models, or just the parameter 
               estimates from the models. 
    kw       : keywords
               keyword arguments passed down to the resulting model estimation statement.
               If for a statsmodels model, should be separated into fit_kw and init_kw. 
    """
    if isinstance(candidate, __sm.base.wrapper.ResultsWrapper):
        out = _sm_jackknife(candidate, **kw)
        if not full:
            out = pd.DataFrame(np.vstack([mod.params.reshape(1,-1) for mod in out]),
                               columns=candidate.params.index)
        return out
    else:
        _pysal_jackknife(candidate, **kw)



def _sm_jackknife(results, init_kw=dict(), fit_kw=dict()):
    """
    Jackknife the results object, meaning the model is reestimated with a single observation left out each time.
    """
    endog, exog = results.model.endog, results.model.exog
    if isinstance(results.model, sm.WLS):
        weights = results.model.weights
    elif isinstance(results.model, sm.OLS):
        weights = np.ones_like(endog)
    deletions = [sm.WLS(np.delete(endog, i),
                        np.delete(exog, i, axis=0),
                        weights=np.delete(weights, i),
                        **init_kw).fit(**fit_kw)
                 for i in range(len(endog))]
    return deletions

def _pysal_jackknife(model, **kw):
    raise NotImplementedError
    X = model.X
    Y = model.Y
    yend = getattr(model, 'yend', None)
    z = getattr(model, 'z', None)
    W = getattr(model, 'W', None)
    fit = type(model)
