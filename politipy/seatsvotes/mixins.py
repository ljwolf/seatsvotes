from warnings import warn as Warn
from .gelmanking import utils as gkutil
import numpy as np
import pandas as pd

class GIGOError(Exception):
    """
    You're trying to do something that will significantly harm
    the validity of your inference.
    """
    pass
    

class Preprocessor(object):
    """
    """
    @classmethod
    def from_arrays(cls, voteshares, 
                       turnout, 
                       years, 
                       redistrict,
                       district_ids,
                       covariates = None,
                       missing = 'drop',
                       uncontested = None,
                       break_on_GIGO = True
                       ):
        """
        Everything must be:
        1. no nan
        2. no uncontesteds/uncontesteds are resolved to 0,1
        3. all arrays are (nt,k), where k is the number of 
            relevant covariates for the attribute. 
        4. 
        """
        if covariates is None:
            covariates = dict()
        frame = pd.DataFrame.from_dict(
                                       dict(vote_share = voteshares,
                                            turnout = turnout,
                                            year = years,
                                            redistrict = redistrict,
                                            district_id = district_ids,
                                            **covariates
                                            ))
        return cls(frame, 
                   share_column='vote_share', 
                   weight_column='turnout',
                   covariates = list(covariates.keys()),
                   years = 'year',
                   redistrict = 'redistrict',
                   district_id = 'district_id',
                   missing=missing,
                   uncontested=uncontested,
                   break_on_GIGO=break_on_GIGO)

    def __init__(self, frame, 
                 share_column='vote_share', 
                 covariates = None,
                 weight_column=None,
                 year_column = None,
                 redistrict_column = None,
                 district_id = 'district_id',
                 missing = 'drop', 
                 uncontested=None,
                 break_on_GIGO=True):
        if break_on_GIGO:
            self._GIGO = lambda x: GIGOError(x)
        else:
            self._GIGO = lambda x: Warn(x, category=GIGOError, stacklevel=2)
        self.elex_frame = frame.copy(deep=True)
        if covariates is None:
            self._covariate_cols = []
        else:
            self._covariate_cols = list(covariates)

        self._year_column =year_column
        self._redistrict_column = redistrict_column
        self._district_id_column = district_id

        if uncontested is None:
            uncontested = dict(method='censor')
        elif isinstance(uncontested, str):
            uncontested = dict(method=uncontested)
        if uncontested['method'].lower().startswith('imp'):
            uncontested['covariates'] = self._covariate_cols
        print(uncontested)
        self._resolve_uncontested(**uncontested)
        if uncontested.get('ordinal', True):
            if uncontested['method'].lower() != 'drop':
                self._covariate_cols.append('uncontested')
        else:
            dummies = pd.get_dummies(self.elex_frame.uncontested)
            dummies.columns = ['uncontested_R', 
                                 'contested', 
                               'uncontested_D']
            self.elex_frame = pd.concat((self.elex_frame, dummies),axis=1)
            self.elex_frame.drop('uncontested', axis=1, inplace=True)
            if uncontested['method'].lower() != 'drop':
                self._covariate_cols.extend(dummies.columns.tolist()) 
        
        self.wide = gkutil.make_designs(self.elex_frame,
                            years=self.elex_frame.get(self._year_column),
                            redistrict=self.elex_frame.get(self._redistrict_column),
                            district_id=self._district_id_column)
        self.long = pd.concat(self.wide, axis=1)

    def _resolve_missing(self, method='drop'):
        if (method.lower() == 'drop'):
            self.elex_frame.dropna(subset=self._covariate_cols,
                                   inplace=True)
        else:
            raise KeyError("Method to resolve missing data not clear."
                           "\n\tRecieved: {}\n\t Supported: 'drop'"
                           "".format(method))

    def _resolve_uncontested(self, method='censor', 
                              floor=None, ceil=None,
                              **special):
        if (method.lower().startswith('winsor') or
            method.lower().startswith('censor')):
            floor, ceil = .25, .75
        elif (method.lower() in ('shift', 'drop')):
            floor, ceil = .05, .95
        elif method.lower().startswith('imp'):
            if special.get('covariates') is []:
                raise self._GIGO("Imputation selected but no covariates "
                                "provided. Shifting uncontesteds to the "
                                "mean is likely to harm the validity "
                                "of inference. Provide a list to "
                                "coviarate_cols to fix.")
            if 'year' not in self.elex_frame:
                raise self._GIGO("Imputation pools over each year. No "
                                "years were provided in the input "
                                "dataframe. Provide a year variate "
                                "in the input dataframe to fix")
        else:
            raise KeyError("Uncontested method not understood."
                            "\n\tRecieved: {}"
                            "\n\tSupported: 'censor', 'winsor', "
                            "'shift', 'drop', 'impute',"
                            " 'impute_recursive'".format(method))
        if self.elex_frame.vote_share.isnull().any():
            raise self._GIGO("There exists a null vote share with full "
                            "covariate information. In order to impute,"
                            "the occupancy of the seat should be known. "
                            "Go through the data and assign records with "
                            "unknown vote share a 0 if the seat was "
                            "awarded to the opposition and 1 if the seat "
                            "was awarded to the reference party to fix.")

        if method.lower() == 'impute_recursive':
            wide = gkutil.make_designs(self.elex_frame,
                            years=self.elex_frame.get(self._year_column),
                            redistrict=self.elex_frame.get(self._redistrict_column),
                            district_id=self._district_id_column)
            design = pd.concat(wide, axis=1)
        else:
            design = self.elex_frame.copy(deep=True)


        self._prefilter = self.elex_frame.copy(deep=True)
        self.elex_frame = _unc[method](design,
                                       floor=floor, ceil=ceil,
                                       **special)

###################################
# Dispatch Table for Uncontesteds #
###################################

def _censor_unc(design, floor=.25, ceil=.75):
    """
    This will clip vote shares to the given mask.
    """
    indicator = ((design.vote_share > ceil).astype(int) + 
                 (design.vote_share < floor).astype(int) * -1)
    design['uncontested'] = indicator
    design['vote_share'] = np.clip(design.vote_share,
                                   a_min=floor, a_max=ceil)
    return design

def _shift_unc(design, floor=.05, ceil=.95, lower_to=.25, ceil_to=.75):
    """
    This replicates the "uncontested.default" method from JudgeIt, which replaces
    the uncontested elections (those outside of the (.05, .95) range) to (.25,.75).
    """
    indicator = ((design.vote_share > ceil).astype(int) + 
                 (design.vote_share < floor).astype(int) * -1)
    design['uncontested'] = indicator
    lowers = design.query('vote_share < @floor').index
    ceils = design.query('vote_share > @ceil').index
    design.ix[lowers, 'vote_share'] = lower_to
    design.ix[ceils, 'vote_share'] = ceil_to
    return design

def _winsor_unc(design, floor=.25, ceil=.75):
    """
    This winsorizes vote shares to a given percentile.
    """
    indicator = ((design.vote_share > ceil).astype(int) + 
                 (design.vote_share < floor).astype(int) * -1)
    design['uncontested'] = indicator
    try:
        from scipy.stats.mstats import winsorize
    except ImportError:
        Warn('Cannot import scipy.stats.mstats.winsorize, censoring instead.',
                stacklevel=2)
        return _censor_unc(shares, floor=floor, ceil=ceil)
    # WARNING: the winsorize function here is a little counterintuitive in that
    #          it requires the ceil limit to be stated as "from the right,"
    #          so it should be less than .5, just like "floor"
    design['vote_share'] = np.asarray(winsorize(design.vote_share, 
                                                limits=(floor, 1-ceil)))
    return design

def _drop_unc(design, floor=.05, ceil=.95):
    """
    This drops uncontested votes that are more extreme than the provided
    floor or ceil.
    """
    design['uncontested'] = 0
    mask = (design.vote_share < floor) + (design.vote_share > (ceil))
    return design[~mask]

def _impute_unc(design, covariates, floor=.25, ceil=.75, fit_params=dict()):
    """
    This imputes the uncontested seats according 
    to the covariates supplied for the model. Notably, this does not
    use the previous years' voteshare to predict the imputed voteshare.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        warn("Must have statsmodels installed to conduct imputation", 
             category=ImportError, stacklevel=2)
        raise
    indicator = ((design.vote_share > ceil).astype(int) + 
                 (design.vote_share < floor).astype(int) * -1)
    design['uncontested'] = indicator
    imputed = []
    for yr,contest in design.groupby("year"):
        mask = (contest.vote_share < floor) | (contest.vote_share > (ceil))
        contested = contest[~mask]
        uncontested = contest[mask]
        unc_ix = uncontested.index
        imputor = sm.WLS(contested.vote_share, 
                         sm.add_constant(contested[covariates]),
                         weights=contested.turnout).fit(**fit_params)
        contest.ix[unc_ix, 'vote_share'] = imputor.predict(
                                                           sm.add_constant(
                                                           uncontested[covariates],
                                                           has_constant='add'))
        imputed.append(contest)
    return pd.concat(imputed, axis=0)

def _impute_using_prev_voteshare(design, covariates,
                                 floor=.25, ceil=.75, fit_params=dict()):
    raise NotImplementedError('recursive imputation, using previous year voteshares to impute '
                              'current uncontested elections, is not supported at this time')
    try:
        import statsmodels.api as sm
    except ImportError:
        warn("Must have statsmodels installed to conduct imputation", 
             category=ImportError, stacklevel=2)
    imputed = []
    covariates += ['vote_share__prev']
    for yr, contest in design.groupby("year"):
        # iterate through, estimating models with only 
        # mutually-contested pairs with no redistricting, then 
        # predict the uncontested election. With that h, move to the next time.
        ...

_unc = dict(censor=_censor_unc,
            shift=_shift_unc,
            judgeit=_shift_unc,
            winsor=_winsor_unc,
            drop=_drop_unc,
            impute=_impute_unc,
            imp=_impute_unc,
            impute_recursive=_impute_using_prev_voteshare)
