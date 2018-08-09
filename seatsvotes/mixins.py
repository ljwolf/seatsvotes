from warnings import warn as Warn
from collections import OrderedDict
from .gelmanking import utils as gkutil
from scipy.optimize import minimize_scalar
import numpy as np
import copy
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
    This exception will tell you that. 
    """
    pass

def _raiseGIGO(msg):
    raise GIGOError(msg)

class Preprocessor(object):
    """
    """
    @classmethod
    def from_arrays(cls, voteshares,
                    turnout,
                    years,
                    redistrict,
                    district_ids,
                    covariates=None,
                    missing='drop',
                    uncontested=None,
                    break_on_GIGO=True
                    ):
        """
        Method mimicking default Preprocessor() construction, but
        with more limited functionality to construct directly from arrays. 
        Specifically, this has less input checking, so to avoid GIGO errors, 
        all input arrays must:
        1. have no nan observations
        2. have no uncontested elections, or at least uncontesteds are resolved to 0,1 and assumed correct.
        3. all arrays are long (n*t,k) where k is the number or relevant covariates for the attribute.

        refer to Preprocessor for more information. 
        """
        if covariates is None:
            covariates = dict()
        frame = pd.DataFrame.from_dict(
                                       dict(vote_share=voteshares,
                                            turnout=turnout,
                                            year=years,
                                            redistrict=redistrict,
                                            district_id=district_ids,
                                            **covariates
                                            ))
        return cls(frame,
                   share_column='vote_share',
                   weight_column='turnout',
                   covariates=list(covariates.keys()),
                   years='year',
                   redistrict='redistrict',
                   district_id='district_id',
                   missing=missing,
                   uncontested=uncontested,
                   break_on_GIGO=break_on_GIGO)

    def __init__(self, frame,
                 share_column='vote_share',
                 covariates=None,
                 weight_column=None,
                 year_column='year',
                 redistrict_column=None,
                 district_id='district_id',
                 missing='drop',
                 uncontested=None,
                 break_on_GIGO=True):
        """
        frame               : pd.DataFrame
                              long dataframe in which the elections data is stored.
        share_column        : str
                              column of dataframe for which the two-party vote shares are stored. 
        covariates          : list of str
                              columns in `frame` to use to predict `share_column`. 
        weight_column       : str
                              turnout or other weight to assign each district in computing weighted 
                              averages for vote shares

        year_column         : str
                              name of column holding the year in which each contest occurs
        redistrict_column   : str
                              name of column holding information about when redistricting occurs 
                              in the dataset. Should be a binary indicator or a boolean 
                              (True if redistricting occurred between that row's `year` and the previous year)
        district_id         : str
                              name of column which contains the district's unique id (stable over years)
        missing             : str
                              missing policy. Can be either:
                                    drop    :   drop entries that are missing any data
                                    impute  :   take the mean of the column as the value in the data
                                    ignore  :   ignore the missing values
                                    (default: drop)
        uncontested         : str
                              uncontested policy. Can be either:
                              censor  :   clip the data to a given vote share
                              winsor  :   clip the data to a given percentage
                              drop    :   drop uncontested elections
                              impute  :   impute vote shares from the available data in each year
                                          on other vote shares
                              impute_recursive: impute vote shares from available data in each year
                                                and the previous year's (possibly imputed) vote share.
                              impute_singlepass: impute vote shares from available data in each year and
                                                 the previous year's vote share. Imputations are not carried forward.
                              (default : censor)

        break_on_GIGO       : bool
                              whether or not to break or proceed when a computation is determined 
                              to yield meaningless but possibly non-null results. This may occur when
                              imputation is requested but covariates are not provided, when simulations
                              are requested for data with incorrect scope or structure, among others. 

                              NOTE: this only catches a few common structural issues in imputation,
                              simulation, and prediction. It does not guarantee that results are "valid."
                              DO NOT change this unless you are sure you know why you need to change this. 
        """
        super().__init__()
        if break_on_GIGO:
            self._GIGO = _raiseGIGO
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
            share_column : 'vote_share',
            district_id : 'district_id',
            year_column : 'year',
            weight_column : 'weight',
            redistrict_column : 'redistrict'
            }, inplace=True)
        try:
            assert len(self.elex_frame.columns) == len(set(self.elex_frame.columns))
        except AssertionError:
            raise AssertionError('Election frame contains duplicated columns: {}'.format(self.elex_frame.columns))
        if weight_column is None:
            self.elex_frame['weight'] = 1

        if uncontested is None:
            uncontested = dict(method='censor')
        elif isinstance(uncontested, str):
            uncontested = dict(method=uncontested)
        if (uncontested['method'].lower().startswith('imp') or 
            uncontested['method'].lower() in ('recursive','singlepass')):
            uncontested['covariates'] = copy.deepcopy(self._covariate_cols)
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


        self.wide = gkutil.make_designs(self.elex_frame,
                            years=self.elex_frame.year,
                            redistrict=self.elex_frame.get('redistrict'),
                            district_id='district_id')
        self.long = pd.concat(self.wide, axis=0)

    def _impute_turnout_from_voteshare_and_state(self, df, turnout_col='turnout',
                                                 state_col='state'):
        """
        Impute turnout from the voteshare and state category. This specifies:
        turnout ~ vote_share + vote_share**2 + StateFixedEffect
        """
        complete = df.dropna(subset=(turnout_col), inplace=False)
        missing_data = df[['vote_share'] + [state_col]].isnull().any(axis=0)
        if missing_data.any():
            missing_cols = df.columns[missing_data]
            self._GIGO("Missing data in imputation of turnout "
                       "for column: {}".format(missing_cols))
        import statsmodels.formula.api as smf, statsmodels.api as sm
        model = smf.OLS('turnout ~ vote_share + I(vote_share**2) + C(state, Treatment)',
                        data=df).fit()
        incomplete_ix = complete[complete[turnout_col].isnull()].index
        incomplete_X = df.ix[incomplete_ix, [turnout_col, state_col]]
        preds = model.predict(sm.add_constant(incomplete_X))
        df.ix[turnout_col, incomplete_ix] = preds
        return df[turnout_col]

    def _resolve_missing(self, method='drop'):
        """
        Resolve missing data issues using a given method:
        drop    :   drop entries that are missing any data
        impute  :   take the mean of the column as the value in the data
        ignore  :   ignore the missing values
        """
        targets = self._covariate_cols + ['weight']
        if (method.lower() == 'drop'):
            self.elex_frame.dropna(subset=targets, inplace=True)
        elif (method.lower() == 'impute'):
            for i, year in self.elex_frame.groupby('year'):
                colmeans = year[targets].mean(axis=0)
                self.elex_frame.ix[year.index, targets] = year[targets].fillna(colmeans)
        elif (method.lower() == 'ignore'):
            return
        else:
            raise KeyError("Method to resolve missing data not clear."
                           "\n\tRecieved: {}\n\t Supported: 'drop'"
                           "".format(method))
        any_in_column = self.elex_frame[targets].isnull().any(axis=0)
        if any_in_column.any():
            still_missing = self.elex_frame.columns[any_in_column]
            self._GIGO('After resolving missing data using {}, the following columns '
                       'still have missing data: {}'.format(still_missing))

    def _resolve_uncontested(self, method='censor',
                              floor=None, ceil=None,
                              **special):
        """
        Resolve uncontested elections' vote shares using a specific method
        censor  :   clip the data to a given vote share
        winsor  :   clip the data to a given percentage
        drop    :   drop uncontested elections
        impute  :   impute vote shares from the available data in each year
                    on other vote shares
        impute_recursive: impute vote shares from available data in each year
                          and the previous year's (possibly imputed) vote share.
        impute_singlepass: impute vote shares from available data in each year and
                           the previous year's vote share. Imputations are not carried forward.
        """
        if method.lower() == 'singlepass':
            method = 'impute_singlepass'
        if method.lower() == 'recursive':
            method = 'impute_recursive'
        if (method.lower().startswith('winsor') or
            method.lower().startswith('censor')):
            floor, ceil = .1,.9
        elif (method.lower() in ('shift', 'drop')):
            floor, ceil = .05, .95
        elif method.lower().startswith('imp'):
            if special.get('covariates') == [] or special.get('covariates') is None:
                self._GIGO("Imputation selected but no covariates "
                                "provided. Shifting uncontesteds to the "
                                "mean is likely to harm the validity "
                                "of inference. Provide a list to "
                                "coviarate_cols to fix.")
            if 'year' not in self.elex_frame:
                self._GIGO("Imputation pools over each year. No "
                                "years were provided in the input "
                                "dataframe. Provide a year variate "
                                "in the input dataframe to fix")
            floor, ceil = .01,.99
            if method.endswith('recursive') or method.endswith('singlepass'):
                # to do the stronger imputation, you need to get the redistricting vector
                if self.elex_frame.get('redistrict') is None:
                    Warn('computing redistricting from years vector')
                    self.elex_frame['redist'] = gkutil.census_redistricting(pd.Series(self.elex_frame.year))
        elif method.lower() == 'ignore':
            floor, ceil = .05, .95
            self.elex_frame['uncontested'] = ((self.elex_frame.vote_share > ceil).astype(int) 
                                     + (self.elex_frame.vote_share < floor).astype(int)*-1)
            return
        else:
            raise KeyError("Uncontested method not understood."
                           "Recieved: {}"
                           "Supported: 'censor', 'winsor', "
                           "'shift', 'drop', 'impute',"
                           " 'impute_recursive', 'impute_singlepass',"
                           "'singlepass'".format(method))
        #if self.elex_frame.vote_share.isnull().any():
        #    raise self._GIGO("There exists a null vote share with full "
        #                    "covariate information. In order to impute,"
        #                    "the occupancy of the seat should be known. "
        #                    "Go through the data and assign records with "
        #                    "unknown vote share a 0 if the seat was "
        #                    "awarded to the opposition and 1 if the seat "
        #                    "was awarded to the reference party to fix.")
        design = self.elex_frame.copy(deep=True)


        self._prefilter = self.elex_frame.copy(deep=True)
        self.elex_frame = _unc[method](design,
                                       floor=floor, ceil=ceil,
                                       **special)

    def _extract_data(self, t=-1, year=None):
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

    def _extract_data_in_model(self, t=-1, year=None):
        """
        Extract an election from the models
        """
        if year is not None:
            t = list(self.years).index(year)
        obs_refparty_shares = self.models[t].model.endog[:,None]
        obs_vote_shares = np.hstack((obs_refparty_shares, 1-obs_refparty_shares))
        obs_seats = (obs_refparty_shares > .5).astype(int)
        obs_turnout = self.models[t].model.weights
        obs_party_vote_shares = np.average(obs_vote_shares, weights=obs_turnout, axis=0)
        obs_party_seat_shares = np.mean(obs_seats, axis=0)

        return (obs_turnout, obs_vote_shares, obs_party_vote_shares, 
                obs_seats, obs_party_seat_shares)

    def _extract_election(self, t=-1, year=None):
        return self._extract_data_in_model(t=t,year=year)

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

    def _extract_data(self, *args, **kwargs):
        raise NotImplementedError("'_extract_data' must be implemented on child class {}"
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
        turnout, vshares, pvshares, *rest = self._extract_data(t=t, year=year)
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

    def plot_simulated_seatsvotes(self, n_sims=10000, swing=0, Xhyp=None,
                                  target_v=None, t=-1, year=None, predict=False,
                                  ax=None, fig_kw=dict(),
                                  scatter_kw=dict(),
                                  mean_center=True, normalize=True,
                                  silhouette=True,
                                  q=[5,50,95],
                                  band=False,
                                  env_kw=dict(), median_kw=dict(),
                                  return_sims=False):
        """
        This plots the full distribution of rank-votes for simulated voteshares.

        Arguments
        ---------
        n_sims
        swing
        Xhyp
        target_v
        t
        year
        predict
        ax
        fig_kw
        scatter_kw
        mean_center
        normalize
        silhouette
        band
        q
        env_kw
        median_kw
        return_sims
        """
        from scipy.stats import rankdata
        if year is not None:
            t = list(self.years).index(year)
        sims = self.simulate_elections(t=t, year=year, n_sims=n_sims, swing=swing,
                                       target_v=target_v, fix=False, predict=predict)
        ranks = [rankdata(1-sim, method='max').astype(float) for sim in sims]
        N = len(sims[0])

        if ax is None:
            f = plt.figure(**fig_kw)
            ax = plt.gca()
        else:
            f = plt.gcf()

        if mean_center:
            target_v = np.average(self.wide[t].vote_share,
                                  weights=self.wide[t].weight)

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
            if band:
                env_kw['alpha']=.4
                ax.fill_betweenx(np.arange(1, N+1)/rescale,
                                 (1-lo)+shift, (1-hi)+shift, **env_kw)
            else:
                ax.plot((1-lo)+shift, np.arange(1, N+1)/rescale, **env_kw)
                ax.plot((1-med)+shift, np.arange(1, N+1)/rescale, **median_kw)
            ax.plot((1-med)+shift, np.arange(1, N+1)/rescale, **median_kw)
        if return_sims:
            return f,ax, sims, ranks
        return f,ax

class AlwaysPredictPlotter(Plotter):
    def plot_simulated_seatsvotes(self, n_sims=10000, swing=0, Xhyp=None,
                                  target_v=None, t=-1, year=None,
                                  ax=None, fig_kw=dict(), predict=True,
                                  scatter_kw=dict(),
                                  mean_center=True, normalize=True,
                                  silhouette=True,
                                  q=[5,50,95],
                                  band=False,
                                  env_kw=dict(), median_kw=dict(),
                                  return_sims=False):
        if predict is False:
            self._GIGO("Prediction should always be enabled for {}".format(self.__class__))
        return Plotter.plot_simulated_seatsvotes(**vars())

class AdvantageEstimator(object):
    ...

class TwoPartyEstimator(AdvantageEstimator):

    def responsiveness(self, hinge = None, 
                       bin_width = .02, **simulation_kws):
        """
        Estimate responsiveness at `hinge` using a linear regression
        over seat shares & vote shares that fall within hinge+/bin_width.

        See `simulate_elections` for more information on simulation options.

        Arguments
        ---------
        hinge    : float
                   vote share (between 0 and 1) around which to estimate 
                   the responsiveness.
        bin_width: float
                   span around the hinge to use in estimating the seats-votes curve
                   over simulations. 
        simulation_kws: keyword arguments
                        see simulate_elections for options. 
        
        Returns:
        --------
        (resp, stderr), where resp is the responsiveness estimate expressed in 
        terms of the impact of a change of a single % of vote share on the expected
        % seat share. 
        """
        assert self._twoparty, 'Default Estimation only currently well-defined for two-party elections'
        simulation_kws.setdefault('target_v', hinge)
        simulation_kws['fix'] = False

        simulations = self.simulate_elections(**simulation_kws)
        if isinstance(simulations, tuple):
            simulations, turnout = simulations
        else:
            turnout_empirical,*_ = self._extract_election(simulation_kws.get('t', -1))
            turnout = np.ones_like(simulations) * turnout_empirical

        from statsmodels import api as sm

        seatshare = (simulations > .5).mean(axis=1)
        voteshare = np.average(simulations, turnout, axis=1)
        in_bin = np.logical_and((voteshare > (hinge - bin_width)),
                                (voteshare < (hinge + bin_width)))
        candidate_voteshares = voteshare[in_bin]
        candidate_seatshares = seatshare[in_bin]
        model = sm.OLS(candidate_seatshares, sm.add_constant(candidate_voteshares))
        _, resp = model.params
        _, stderr = model.bse
        return resp*.01, stderr*.01

    def winners_bonus(self, hinge = None 
                      q = [2.5, 50, 97.5], **simulation_kws):
        """
        estimate the winners bonus at a known hinge. Common hinge
        values are the empirical mean vote and 50% (.5). Defaults 
        to the empirical average. This is also known as "partisan bias"
        from Gelman & King (1994), by the partisan symmetry standard. 

        Arguments
        ----------
        hinge    : float
                   vote share (between 0 and 1) around which to estimate 
                   the responsiveness.
        q        : iterable (default: [2.5, 50, 97.5], the 95% simulation interval)
                    percentiles to pass to numpy.percentile to extract from the
                    distribution of simulated mean-median differences.
                    If None, all simulated median-mean divergences are returned. 
        simulation_kws: keyword arguments
                        see simulate_elections for options.

        """
        assert self._twoparty, 'Default Estimation only currently well-defined for two-party elections'
        if hinge is None:
            results = self._extract_election(simulation_kws.get('t', -1))
            hinge = np.average(results[2])
        simulation_kws['target_v'] = hinge
        sims_A = self.simulate_elections(**simulation_kws)
        simulation_kws['target_v'] = 1 - hinge
        sims_B = self.simulate_elections(**simulation_kws)
        if isinstance(sims_A, tuple):
            sims_A, turnout_A = sims_A
            sims_B, turnout_B = sims_B
        else:
            turnout_A = turnout_B = np.ones_like(sims_A) * results[1]

        party_seatshares_A = (sims_A > .5).mean(axis=1)
        party_seatshares_B = (sims_B > .5).mean(axis=1)

        opponent_seatshares_B = 1 - party_seatshares_B
        if q is None:
            return party_seatshares_A - opponent_seatshares_B
        return np.percentile(party_seatshares_A - opponent_seatshares_B,
                             q=q)

    def optimal_attainment_gap(self, 
                               n_batches = 1000,
                               batch_size = 100, 
                               loss = 'mad',
                               q = [2.5, 50, 97.5]
                               **simulation_kws):
        """
        estimate the expected minimum average vote share needed to obtain a 
        legislative majority for either party. 

        Arguments
        ---------
        n_batches : int
                    how many times to run the stochastic optimization algorithm
                    to obtain expected minimum voteshares
        batch_size: int
                    how large each stochastic optimization's simulation batches
                    should be.
        loss      : str or callable (default: 'mad')
                    if string, should be either 'mad' (mean absolute deviation)
                    or 'ssd' (sum of squared differences). Otherwise, this should
                    be a callable convex loss function that takes an array and returns
                    a single scalar computing how far off the target seat share
                    is from the desired seat share (should be .5 for most applications)
        q         : iterable or None
                    iterable of percentiles passed to numpy.percentile to summarize the 
                    observed & opposition minimum voteshares. If None, returns all
                    simulated minima for both parties. 
        simulation_kws: keyword arguments
                        see simulate_elections for options.
        
        Returns
        -------
        tuple of expected smallest majority-winning vote-shares,
        summarized at percentiles in `q`: (reference_percentiles, opponent_percentiles)
        if q is none, then all simulations are returned for both reference & opponent.
        Further, if q is None, a differing number of simulations can be obtained 
        for reference & opponent party if the stochastic optimization does not converge
        in any given batch. 
        """
        assert self._twoparty, 'Default Estimation only currently well-defined for two-party elections'
        simulation_kws['n_sims'] = batch_size
        if loss == 'mad':
            loss = lambda x: np.abs(x - .5).mean()
        elif loss == 'ssd':
            loss = lambda x: np.sum((x - .5)**2)
        elif not callable(loss):
            raise ValueError("Unrecognized loss function: {}".format(loss))
        def seatgap(target):
            simulation_kws['target_v'] = target
            sims = self.simulate_elections(**simulation_kws)
            if isinstance(sims, tuple):
                sims, _ = sims
            seatshares = (sims > .5).mean(axis=1)
            return loss(seatshares)
        def oppo_seatgap(target):
            return seatgap(1 - target)
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(a, desc=''):
                return a
        best_targets = []
        best_oppos = []
        for _ in tqdm(range(n_batches), desc='optimizing batches...'):
            best_targets.append(minimize_scalar(seatgap,
                                                tol = 1e-4,
                                                bounds=(.05,.95), 
                                                method='bounded'))
            best_oppos.append(minimize_scalar(oppo_seatgap, 
                                              tol = 1e-4,
                                              bounds=(.05,.95), 
                                              method='bounded'))
        best_references = np.asarray([op.x for op in best_targets if op.success])
        best_oppos = np.asarray([op.x for op in best_oppos if op.success])
        if q is None:
            return best_references, best_oppos
        else:
            q_ref = np.percentile(best_references, q=q)
            q_opp = np.percentile(best_oppos, q=q)
            return q_ref, q_opp

    def efficiency_gap(self, q=[2.5, 50, 97.5], 
                       use_empirical_turnout=True, **simulation_kws):
        """
        estimate the efficiency gap over simulated elections
        """
        assert self._twoparty, 'Default Estimation only currently well-defined for two-party elections'
        t = simulation_kws.get('t', -1)
        empirical_turnout, vote_shares, party_vote_share, *_ = self._extract_election(t)
        simulation_kws.setdefault('target_v', party_vote_share)
        from .estimators import efficiency_gap
        sims = self.simulate_elections(**simulation_kws)
        if isinstance(sims, tuple):
            sims, turnout = sims
        else:
            turnout = np.ones_like(sims)
            if use_empirical_turnout:
                turnout *= empirical_turnout
        gaps = [efficiency_gap(elex, turnout_i) for
                 elex, turnout in zip(sims,turnout)]
        if q is None:
            return gaps
        else:
            return np.percentile(gaps, q=q)

    def median_mean_divergence(self, q=[2.5, 50, 97.5], **simulation_kws):
        """
        estimate the divergence between median and mean district
        vote shares in simulations, from McDonald & Best (2016). 
        Arguments
        ---------
        q       :   iterable (default: [2.5, 50, 97.5], the 95% simulation interval)
                    percentiles to pass to numpy.percentile to extract from the
                    distribution of simulated mean-median differences.
                    If None, all simulated median-mean divergences are returned. 
        simulation_kws: keyword arguments
                        see simulate_elections for options.
        Returns
        -------
        percentiles according to q or, if q is None, all simulations.
        """
        assert self._twoparty, 'Default Estimation only currently well-defined for two-party elections'
        simulations = self.simulate_elections(**simulation_kws)
        if isinstance(simulations, tuple):
            simulations, turnout = simulations
        else:
            t = simulation_kws.get('t', -1)
            turnout = np.ones_like(simulations) * self._extract_election(t)[0]
        mmds = np.median(simulations, axis=1) - np.average(simulations,
                                                           weights=turnout,
                                                           axis=1)
        if q is None:
            return mmds
        else:
            return np.percentile(mmds, q=q)



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
        return _censor_unc(design, floor=floor, ceil=ceil)
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


def _impute_unc(design, covariates,floor=.25, ceil=.75, fit_params=dict()):
    """
    This imputes the uncontested seats according
    to the covariates supplied for the model. Notably, this does not
    use the previous years' voteshare to predict the imputed voteshare.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        Warn("Must have statsmodels installed to conduct imputation",
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
                         sm.add_constant(contested[covariates], has_constant='add'),
                         weights=contested.weight).fit(**fit_params)
        contest.ix[unc_ix, 'vote_share'] = imputor.predict(
                                                           sm.add_constant(
                                                           uncontested[covariates],
                                                           has_constant='add'))
        imputed.append(contest)
    return pd.concat(imputed, axis=0)

def _impute_singlepass(design, covariates, floor=.01, ceil = .99, fit_params=dict()):
    """
    Impute the uncontested vote shares using a single-pass strategy. This means that
    a model is fit on mutually-contested elections in each year, and then elections
    that are uncontested are predicted out of sample. Critically, imputed values 
    are *not* propagated forward, so that imputation in time t does not affect estimates
    for t+1.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        Warn("Must have statsmodels installed to conduct imputation",
                category=ImportError, stacklevel=2)
        raise
    indicator = ((design.vote_share > ceil).astype(int) + 
                  (design.vote_share < floor).astype(int) * -1)
    design['uncontested'] = indicator
    wide = gkutil.make_designs(design,
                               years=design.year,
                               redistrict=design.get('redistrict'),
                               district_id='district_id')
    results = []
    for i, elex in enumerate(wide):
        uncontested = elex.query('vote_share in (0,1)')
        contested = elex[~elex.index.isin(uncontested.index)]
        covs = copy.deepcopy(covariates)
        if 'vote_share__prev' in elex.columns:
            covs.append('vote_share__prev')
        X = contested[covs].values
        Xc = sm.add_constant(X, has_constant='add')
        Y = contested[['vote_share']]
        model = sm.WLS(endog=Y, exog=Xc, weights=contested.weight,
                       missing='drop').fit(**fit_params)
        OOSXc = sm.add_constant(uncontested[covs].values, has_constant='add')
        out = model.predict(OOSXc)
        elex.ix[uncontested.index, 'vote_share'] = out
        results.append(elex)
    results = pd.concat(results, axis=0)
    results.drop('vote_share__prev', axis=1, inplace=True)
    return results

def _impute_recursive(design, covariates,floor=.01, ceil=.99, fit_params=dict()):
    """
    This must iterate over each year, fit a model on that year and last 
    year (if available), and then predict that years' uncontesteds.
    Then, it must propagate these predictions forward. 
    """
    #design['vote_share__original'] = design['vote_share']
    ## we're gonna fit models
    try:
        import statsmodels.api as sm
    except ImportError:
        Warn("Must have statsmodels installed to conduct imputation",
             category=ImportError, stacklevel=2)
        raise
    #use the same strategy of the uncontested variate as before
    covariates += ['vote_share__prev']
    indicator = ((design.vote_share > ceil).astype(int) +
                 (design.vote_share < floor).astype(int) * -1)
    design['uncontested'] = indicator
    grouper = iter(design.groupby('year'))
    out = []
    imputers = []
    last_year, last_data = next(grouper)
    last_data = _impute_unc(last_data, covariates=covariates[:-1],
                            floor=floor, ceil=ceil, **fit_params)
    out.append(last_data.copy(deep=True))
    for yr, contest in grouper:
        if 'vote_share__prev' in contest.columns:
            contest.drop('vote_share__prev', inplace=True, axis=1)
        if 'vote_share__prev' in last_data.columns:
            last_data.drop('vote_share__prev', inplace=True, axis=1)
        assert 'vote_share__prev' not in contest.columns
        assert len(contest.columns) == len(set(contest.columns))
        contest = contest.merge(last_data[['district_id','vote_share']],
                                on='district_id', suffixes=('','__prev'),
                                how='left')
        if contest.vote_share__prev.isnull().all():
            raise GIGOError('No match between two panels found in {}. Check that'
                            ' the district_id is correctly specified, in that it'
                            ' identifies districts uniquely within congresses, '
                            ' and can be used to join one year worth of data '
                            ' to another'.format(yr))
        if contest.redist.all():
            #if it's a redistricting cycle, impute like we don't have
            #the previous years' voteshares
            contest = _impute_unc(contest, covariates=covariates[:-1],
                                  floor=floor, ceil=ceil,
                                  **fit_params)
            contest['vote_share__prev'] = np.nan
            out.append(contest.copy(deep=True))
            last_data = contest.copy(deep=True)
            continue
        contested = contest.query('uncontested == 0')
        uncontested = contest[~contest.index.isin(contested.index)]
        assert contested.shape[0] + uncontested.shape[0] == contest.shape[0]
        Xc = sm.add_constant(contested[covariates], has_constant='add')
        model = sm.WLS(contested.vote_share, 
                       Xc, weights=contested.weight,
                       missing='drop'
                       ).fit(**fit_params)
        imputers.append(model)
        Xunc =  sm.add_constant(uncontested[covariates],has_constant='add') 
        predictions = model.predict(Xunc)
        contest.ix[uncontested.index, 'vote_share'] = predictions
        last_data = contest.copy(deep=True)
        out.append(contest.copy(deep=True))
    altogether = pd.concat(out, axis=0)
    altogether.drop('vote_share__prev', axis=1, inplace=True)
    return altogether


# iterate through, estimating models with only
# mutually-contested pairs with no redistricting, then
# predict the uncontested election. With that h, move to the next time.

_unc = dict(censor=_censor_unc,
            shift=_shift_unc,
            judgeit=_shift_unc,
            winsor=_winsor_unc,
            drop=_drop_unc,
            impute=_impute_unc,
            imp=_impute_unc,
            impute_singlepass=_impute_singlepass,
            singlepass=_impute_singlepass,
            impute_recursive=_impute_recursive)
