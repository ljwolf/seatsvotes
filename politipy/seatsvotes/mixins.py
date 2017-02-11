from warnings import warn as Warn
from collections import OrderedDict
from .gelmanking import utils as gkutil
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    class RaiseForMissingMPL(object):
        def __getattribute__(*_, **__):
            raise ImportError("No module named matplotlib") 
    plt = RaiseForMissingMPL()

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
                 year_column = 'year',
                 redistrict_column = None,
                 district_id = 'district_id',
                 missing = 'drop', 
                 uncontested=None,
                 break_on_GIGO=True):
        super().__init__()
        if break_on_GIGO:
            self._GIGO = lambda x: GIGOError(x)
        else:
            self._GIGO = lambda x: Warn(x, category=GIGOError, stacklevel=2)
        self.elex_frame = frame.copy(deep=True)
        if covariates is None:
            self._covariate_cols = []
        else:
            self._covariate_cols = list(covariates)
        provided = [x for x in (share_column, *self._covariate_cols,
                                weight_column,
                                year_column, redistrict_column,
                                district_id) if x is not None]
        self.elex_frame[provided]
        self.elex_frame.rename(columns={
            district_id : 'district_id',
            year_column : 'year',
            weight_column : 'weight',
            redistrict_column : 'redistrict'
            }, inplace=True)
        if weight_column is None:
            self.elex_frame['weight'] = 1

        if uncontested is None:
            uncontested = dict(method='censor')
        elif isinstance(uncontested, str):
            uncontested = dict(method=uncontested)
        if uncontested['method'].lower().startswith('imp'):
            uncontested['covariates'] = self._covariate_cols
        self._resolve_missing(method=missing)
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


        if year_column is not None:
            try:
                self.elex_frame['year'] = self.elex_frame.year.astype(int)
            except KeyError:
                raise KeyError("The provided year column is not found in the dataframe."
                               " Provided: {}".format(self._year_column))
        if redistrict_column is not None:
            try:
                self.elex_frame.redistrict = self.elex_frame.redistrict.astype(int)
            except KeyError:
                raise KeyError("The provided year column is not found in the dataframe."
                               "\n\tProvided: {}".format(self._redistrict_column))
        self.wide = gkutil.make_designs(self.elex_frame,
                            years=self.elex_frame.year,
                            redistrict=self.elex_frame.get('redistrict'),
                            district_id='district_id')
        self.long = pd.concat(self.wide, axis=0)

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
            floor, ceil = .25, .75
        else:
            raise KeyError("Uncontested method not understood."
                            "\n\tRecieved: {}"
                            "\n\tSupported: 'censor', 'winsor', "
                            "'shift', 'drop', 'impute',"
                            " 'impute_recursive'".format(method))
        #if self.elex_frame.vote_share.isnull().any():
        #    raise self._GIGO("There exists a null vote share with full "
        #                    "covariate information. In order to impute,"
        #                    "the occupancy of the seat should be known. "
        #                    "Go through the data and assign records with "
        #                    "unknown vote share a 0 if the seat was "
        #                    "awarded to the opposition and 1 if the seat "
        #                    "was awarded to the reference party to fix.")

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
    
    def _extract_election(self, t=-1, year=None):
        """
        get the essential statistics from the `t`th election.

        Argument
        ---------
        t       :   int
                    index of time desired. This should be a valid index to self.models
        year    :   int
                    index of year desired. This should be some year contained in the index of self.params

        Returns
        ----------
        a tuple of observed data:
        turnout             : (n,1) vector of the turnouts over n districts in election t
        vote_shares         : (n,p) the share of votes won by party j, j = 1, 2, ... p
        party_vote_shares   : (p,)  the share of overall votes won by party j
        seats               : (n,p) the binary indicators showing whether party j won each seat
        party_seat_share    : (p,)  the share of overall seats won by party j
        """
        if year is not None:
            t = list(self.years).index(year)
        obs_refparty_shares = self.wide[t].vote_share[:,None]
        obs_vote_shares = np.hstack((obs_refparty_shares, 1-obs_refparty_shares))
        obs_seats = (obs_vote_shares > .5).astype(int)
        obs_turnout = self.wide[t].weight
        obs_party_vote_shares = np.average(obs_vote_shares,
                                           weights=obs_turnout, axis=0)
        obs_party_seat_shares = np.mean(obs_seats, axis=0)
        return (obs_turnout, obs_vote_shares, obs_party_vote_shares,
                             obs_seats, obs_party_seat_shares)

class Plotter(object):
    """
    Class to proide plotting capabilities to various seats-votes simulation methods. 
    """

    def __init__(self):
        super().__init__()

    @property
    def years(self):
        raise NotImplementedError("'years' must be implemented on child class {}"
                                  "In order to be used.".format(type(self)))

    def _extract_election(self, *args, **kwargs):
        raise NotImplementedError("'_extract_election' must be implemented on child class {}"
                                  " in order to be used.".format(type(self)))
    
    def simulate_elections(self, *args, **kwargs):
        raise NotImplementedError("'simulate_elections' must be implemented on child class {}"
                                  " in order to be used.".format(type(self)))
        

    def plot_rankvote(self, t=-1, year= None, normalize=False, mean_center=False,
                      ax=None, fig_kw = dict(), scatter_kw=dict(c='k')): 
        """
        Plot the rankvote curve for the given time period. 

        Arguments
        ---------
        t   :   int
                time index
        year:   int
                year to plot. Overrides t
        normalize   :   bool
                        flag denoting whether to normalize ranks to [0,1]
        mean_center :   bool
                        flag denoting whether to center the rankvote to the 
                        party vote share. If both normalize and mean_center, 
                        the plot is actually the seats-votes curve.
        ax          :   matplotlib.AxesSubPlot
                        an axis to plot the data on. If None, will create a new
                        figure.
        fig_kw      :   dict
                        keyword arguments for the plt.figure() call, if applicable.
        scatter_kw  :   dict
                        keyword arguments for the ax.scatter call, if applicable.

        Returns
        -------
        figure and axis of the rank vote plot
        """
        from scipy.stats import rankdata 
        turnout, vshares, pvshares, *rest = self._extract_election(t=t, year=year)
        vshares = vshares[:,0]
        if ax is None:
            f = plt.figure(**fig_kw)
            ax = plt.gca()
        else:
            f = plt.gcf()
        ranks = rankdata(1-vshares, method='max').astype(float)
        if normalize:
            ranks = ranks / len(ranks)
        if mean_center:
            plotshares = (1 - vshares) + (pvshares[0] - .5)
        else:
            plotshares = 1 - vshares
        ax.scatter(plotshares, ranks, **scatter_kw)
        if normalize and mean_center:
            title = 'Seats-Votes Curve ({})'
        elif normalize:
            title = 'Normalized Rank-Votes Curve ({})'
        elif mean_center:
            title = 'Centered Rank-Votes Curve ({})'
        else:
            title = 'Rank-Votes Curve ({})'
        if year is None:
            year = self.years[t]
        ax.set_title(title.format(year))
        return f,ax

    def plot_empirical_seatsvotes(self, *args, **kwargs):
        """
        This is plot_rankvote with normalize and mean_center forced to be true.
        """
        kwargs['normalize'] = True
        kwargs['mean_center'] = True
        return self.plot_rankvote(*args, **kwargs)

    def plot_simulated_seatsvotes(self, n_sims=10000, swing=None, Xhyp=None,
                                  target_v=None, t=-1, year=None, predict=False,
                                  ax=None, fig_kw=dict(),
                                  scatter_kw=dict(),
                                  mean_center=True, normalize=True, 
                                  silhouette = True, 
                                  q=[5,50,95], 
                                  env_kw=dict(), median_kw=dict(),
                                  return_sims = False):
        from scipy.stats import rankdata
        if target_v is None:
            target_v = self._extract_election(t=t, year=year)[2][0] #democrat party vote share in t/year
        sims = self.simulate_elections(t=t, year=year, n_sims=n_sims, swing=swing, 
                                       target_v=target_v, fix=False, predict=predict)
        ranks = [rankdata(1-sim, method='max').astype(float) for sim in sims] 
        N = len(sims[0])
        
        if ax is None:
            f = plt.figure(**fig_kw)
            ax = plt.gca()
        else:
            f = plt.gcf()

        shift = (target_v - .5) if mean_center else 0
        rescale = N if normalize else 1

        if silhouette:
            #force silhouette aesthetics
            scatter_kw['alpha'] = scatter_kw.get('alpha', .01)
            scatter_kw['color'] = scatter_kw.get('color', 'k')
            scatter_kw['linewidth'] = scatter_kw.get('linewidth', 0)
            scatter_kw['marker'] = scatter_kw.get('marker', 'o')
            tally = OrderedDict()
            tally.update({i:[] for i in range(1, N+1)})
            for sim, rank in zip(sims, ranks):
                for hi, ri in zip(sim, rank):
                    tally[ri].append(hi)
            ptiles = OrderedDict([(i,np.percentile(tally[i], q=q)) for i in tally.keys()])
            lo, med, hi = np.vstack(ptiles.values()).T
        else:
            #suggest these otherwise, if user doesn't provide alternatives
            scatter_kw['alpha'] = scatter_kw.get('alpha', .2)
            scatter_kw['color'] = scatter_kw.get('color', 'k')
            scatter_kw['linewidth'] = scatter_kw.get('linewidth', 0)
            scatter_kw['marker'] = scatter_kw.get('marker', 'o')
        for sim, rank in zip(sims, ranks):
            ax.scatter((1-sim)+shift, rank/rescale, **scatter_kw)
        if silhouette:
            env_kw['linestyle'] = env_kw.get('linestyle', '-')
            env_kw['color'] = env_kw.get('color', '#FD0E35')
            env_kw['linewidth'] = env_kw.get('linewidth', 1)
            median_kw['linestyle'] = median_kw.get('linestyle', '-')
            median_kw['color'] = median_kw.get('color', '#FD0E35')
            ax.plot((1-lo)+shift, np.arange(1, N+1)/rescale, **env_kw)
            ax.plot((1-med)+shift, np.arange(1, N+1)/rescale, **median_kw)
            ax.plot((1-hi)+shift, np.arange(1, N+1)/rescale, **env_kw)
        if return_sims:
            return f,ax, sims, ranks
        return f,ax

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

def _impute_unc(design, covariates, 
                floor=.25, ceil=.75, fit_params=dict()):
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
        mask |= contest.vote_share.isnull()
        contested = contest[~mask]
        uncontested = contest[mask]
        unc_ix = uncontested.index
        imputor = sm.WLS(contested.vote_share, 
                         sm.add_constant(contested[covariates]),
                         weights=contested.weight).fit(**fit_params)
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
