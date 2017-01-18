import numpy as np
import statsmodels.api as sm
from . import utils as ut
import pandas as pd


def models(designs, share_col, covariate_cols, weight_col=None):
    return [_model(design, share_col, covariate_cols, weight_col=weight_col)
            for design in designs]


def _model(design, share_col, covariate_cols, weight_col=None):
    if weight_col is not None:
        weights = 1.0 / design[weight_col].values
    else:
        weights = 1
    last_years = share_col + '__prev'
    if last_years in design.columns:
        data = design[covariate_cols + [last_years]]
    else:
        data = design[covariate_cols]
    data = sm.add_constant(data, prepend=True, has_constant='add')
    response = design[[share_col]]
    model = sm.WLS(response, data, weights=weights, missing='drop').fit()
    return model
