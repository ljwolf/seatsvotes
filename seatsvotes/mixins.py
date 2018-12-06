from warnings import warn as Warn
from collections import OrderedDict
from . import utils
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
                              ignore  :   do nothing about uncontested elections.
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
        self.elex_frame = frame.copy()
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
            share_column: 'vote_share',
            district_id: 'district_id',
            year_column: 'year',
            weight_column: 'weight',
            redistrict_column: 'redistrict'
        }, inplace=True)
        try:
            assert len(self.elex_frame.columns) == len(
                set(self.elex_frame.columns))
        except AssertionError:
            raise AssertionError('Election frame contains '
                                 'duplicated columns: {}'.format(
                                     self.elex_frame.columns))
        if weight_column is None:
            self.elex_frame['weight'] = 1

        if uncontested is None:
            uncontested = dict(method='censor')
        elif isinstance(uncontested, str):
            uncontested = dict(method=uncontested)
        if (uncontested['method'].lower().startswith('imp') or
                uncontested['method'].lower() in ('recursive', 'singlepass')):
            uncontested['covariates'] = copy.deepcopy(self._covariate_cols)
        if year_column is not None:
            try:
                self.elex_frame['year'] = self.elex_frame.year.astype(int)
            except KeyError:
                raise KeyError("The provided year column is not "
                               "found in the dataframe."
                               " Provided: {}".format(self._year_column))
        if redistrict_column is not None:
            try:
                self.elex_frame.redistrict = self.elex_frame.redistrict.astype(
                    int)
            except KeyError:
                raise KeyError("The provided year column is "
                               "not found in the dataframe."
                               "\n\tProvided: {}".format(
                                   self._redistrict_column))
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
            self.elex_frame = pd.concat((self.elex_frame, dummies), axis=1)
            self.elex_frame.drop('uncontested', axis=1, inplace=True)
            if uncontested['method'].lower() != 'drop':
                self._covariate_cols.extend(dummies.columns.tolist())

        self.wide = utils.make_designs(self.elex_frame,
                                       years=self.elex_frame.year,
                                       redistrict=self.elex_frame.get(
                                           'redistrict'),
                                       district_id='district_id')
        self.long = pd.concat(self.wide, axis=0, sort=True)

    @staticmethod
    def _impute_turnout_from_voteshare_and_state(df, turnout_col='turnout',
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
                       "for column: {}".format(missing_data))
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
        model = smf.OLS('turnout ~ vote_share + '
                        ' I(vote_share**2) + C(state, Treatment)',
                        data=df).fit()
        incomplete_ix = complete[complete[turnout_col].isnull()].index
        imcomplete_X = df.ix[incomplete, [turnout_col, state_col]]
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
                self.elex_frame.ix[year.index,
                                   targets] = year[targets].fillna(colmeans)
        elif (method.lower() == 'ignore'):
            return
        else:
            raise KeyError("Method to resolve missing data not clear."
                           "\n\tRecieved: {}\n\t Supported: 'drop'"
                           "".format(method))
        any_in_column = self.elex_frame[targets].isnull().any(axis=0)
        if any_in_column.any():
            still_missing = self.elex_frame.columns[any_in_columns]
            self._GIGO('After resolving missing data '
                       'using {}, the following columns '
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
        elif method.lower() == 'recursive':
            method = 'impute_recursive'
        elif (method.lower().startswith('winsor') or
                method.lower().startswith('censor')):
            floor, ceil = .1, .9
        elif (method.lower() in ('shift', 'drop')):
            floor, ceil = .05, .95
        elif method.lower().startswith('imp'):
            if (special.get('covariates') == []
                    or special.get('covariates') is None):
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
            floor, ceil = .01, .99
            if method.endswith('recursive') or method.endswith('singlepass'):
                # to do the stronger imputation, you need to get the redistricting vector
                if self.elex_frame.get('redistrict') is None:
                    Warn('computing redistricting from years vector')
                    self.elex_frame['redist'] = utils.census_redistricting(
                        pd.Series(self.elex_frame.year))
        elif method.lower() == 'ignore':
            floor, ceil = .05, .95
            self.elex_frame['uncontested'] = ((self.elex_frame.vote_share > ceil).astype(int)
                                              + (self.elex_frame.vote_share < floor).astype(int)*-1)
            return
        else:
            raise KeyError("Uncontested method not understood."
                           "Recieved: {}"
                           "Supported: 'censor', 'winsor', "
                           "'shift', 'drop', 'impute', 'ignore',"
                           " 'impute_recursive', 'impute_singlepass',"
                           "'singlepass'".format(method))
        # if self.elex_frame.vote_share.isnull().any():
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
        obs_refparty_shares = self.wide[t].vote_share[:, None]
        obs_vote_shares = np.hstack(
            (obs_refparty_shares, 1-obs_refparty_shares))
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
        obs_refparty_shares = self.models[t].model.endog[:, None]
        obs_vote_shares = np.hstack(
            (obs_refparty_shares, 1-obs_refparty_shares))
        obs_seats = (obs_refparty_shares > .5).astype(int)
        obs_turnout = self.models[t].model.weights
        obs_party_vote_shares = np.average(
            obs_vote_shares, weights=obs_turnout, axis=0)
        obs_party_seat_shares = np.mean(obs_seats, axis=0)

        return (obs_turnout, obs_vote_shares, obs_party_vote_shares,
                obs_seats, obs_party_seat_shares)

    def _extract_election(self, t=-1, year=None):
        return self._extract_data_in_model(t=t, year=year)


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

    def plot_rankvote(self, t=-1, year=None, normalize=False, mean_center=False,
                      ax=None, fig_kw=dict(), scatter_kw=dict(c='k')):
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
        vshares = vshares[:, 0]
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
        return f, ax

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
                                  q=[5, 50, 95],
                                  band=False,
                                  env_kw=dict(), median_kw=dict(),
                                  return_sims=False):
        """
        This plots the full distribution of rank-votes for simulated voteshares.

        Arguments
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
            # force silhouette aesthetics
            scatter_kw['alpha'] = scatter_kw.get('alpha', .01)
            scatter_kw['color'] = scatter_kw.get('color', 'k')
            scatter_kw['linewidth'] = scatter_kw.get('linewidth', 0)
            scatter_kw['marker'] = scatter_kw.get('marker', 'o')
            tally = OrderedDict()
            tally.update({i: [] for i in range(1, N+1)})
            for sim, rank in zip(sims, ranks):
                for hi, ri in zip(sim, rank):
                    tally[ri].append(hi)
            ptiles = OrderedDict(
                [(i, np.percentile(tally[i], q=q)) for i in tally.keys()])
            lo, med, hi = np.vstack(ptiles.values()).T
        else:
            # suggest these otherwise, if user doesn't provide alternatives
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
                env_kw['alpha'] = .4
                ax.fill_betweenx(np.arange(1, N+1)/rescale,
                                 (1-lo)+shift, (1-hi)+shift, **env_kw)
            else:
                ax.plot((1-lo)+shift, np.arange(1, N+1)/rescale, **env_kw)
                ax.plot((1-hi)+shift, np.arange(1, N+1)/rescale, **median_kw)
            ax.plot((1-med)+shift, np.arange(1, N+1)/rescale, **median_kw)
        if return_sims:
            return f, ax, sims, ranks
        return f, ax


class AlwaysPredictPlotter(Plotter):
    def plot_simulated_seatsvotes(self, n_sims=10000, swing=0, Xhyp=None,
                                  target_v=None, t=-1, year=None,
                                  ax=None, fig_kw=dict(), predict=True,
                                  scatter_kw=dict(),
                                  mean_center=True, normalize=True,
                                  silhouette=True,
                                  q=[5, 50, 95],
                                  band=False,
                                  env_kw=dict(), median_kw=dict(),
                                  return_sims=False):
        if predict is False:
            self._GIGO(
                "Prediction should always be enabled for {}".format(self.__class__))
        return Plotter.plot_simulated_seatsvotes(**vars())


class AdvantageEstimator(object):

    @staticmethod
    def _do_statistic(sims, *additional_parameters, **named_params):
        # do advantage algorithm using simulations & knowns explicitly provided
        return

    def statistic(self, *additional_parameters, sim_kws={}, stat_kws={}):
        sims = self.simulate(sim_kws)
        self._do_statistic(self, *additional_parameters, stat_kws)

    def get_swing_ratio(self, n_sims=1000, t=-1,
                        Xhyp=None,
                        predict=False, use_sim_swing=True):
        """
        Generic method to either compute predictive or counterfactual elections.

        See also: predict, counterfactal

        Arguments
        ---------
        n_sims      :   int
                        number of simulations to conduct
        t           :   int
                        the target year to use for the counterfactual simulations
        swing       :   float
                        arbitrary shift in vote means
        Xhyp        :   (n,k)
                        artificial data to use in the simulation
        target_v    :   float
                        target mean vote share to peg the simulations to. Will 
                        ensure that the average of all simulations conducted is
                        this value.
        fix         :   bool
                        flag to denote whether each simulation is pegged exactly
                         to `target_v`, or if it's only the average of all 
                         simulations pegged to this value.
        predict     :   bool
                        whether or not to use the predictive distribution or the
                         counterfactual distribution
        use_sim_swing:  bool
                        whether to use the instantaneous change observed in 
                        simulations around the observed seatshare/voteshare 
                        point, or to use the aggregate slope of the seats-votes
                         curve over all simulations as the swing ratio
        """
        # Simulated elections
        simulations = self.simulate_elections(n_sims=n_sims, t=t,
                                              swing=None, Xhyp=Xhyp,
                                              target_v=.5, fix=False, predict=predict)
        turnout = 1/self.models[t].model.weights
        ref_voteshares = np.average(simulations, weights=turnout, axis=1)
        grand_ref_voteshare = ref_voteshares.mean()

        ref_seatshares = (simulations > .5).mean(axis=1)
        grand_ref_seatshare = ref_seatshares.mean()

        # chose to do this via tuples so that we can use the method elsewhere
        obs_turnout, *rest = self._extract_election(t=t)
        obs_voteshares, obs_party_voteshares, *rest = rest
        obs_seats, obs_party_seatshares = rest

        # Swing Around Median
        party_voteshares = np.hstack((ref_voteshares.reshape(-1, 1),
                                      1-ref_voteshares.reshape(-1, 1)))
        party_seatshares = np.hstack((ref_seatshares.reshape(-1, 1),
                                      1-ref_seatshares.reshape(-1, 1)))

        swing_near_median = est.swing_about_pivot(party_seatshares,
                                                  party_voteshares,
                                                  np.ones_like(obs_party_voteshares)*.5)

        # Swing near observed voteshare
        shift_simulations = simulations + (obs_party_voteshares[0] - .5)
        shift_ref_voteshares = np.average(shift_simulations,
                                          weights=turnout, axis=1)
        shift_ref_seatshares = (shift_simulations > .5).mean(axis=1)

        shift_party_voteshares = np.hstack((shift_ref_voteshares.reshape(-1, 1),
                                            1-shift_ref_voteshares.reshape(-1, 1)))
        shift_party_seatshares = np.hstack((shift_ref_seatshares.reshape(-1, 1),
                                            1-shift_ref_seatshares.reshape(-1, 1)))

        swing_at_observed = est.swing_about_pivot(shift_party_seatshares,
                                                  shift_party_voteshares,
                                                  obs_party_voteshares)
        # Sanity Check
        if not np.isfinite(swing_near_median).all():
            Warn('The computation returned an infinite swing ratio. Returning for'
                 ' debugging purposes...', stacklevel=2)
            return party_seatshares, party_voteshares, obs_party_voteshares
        elif not np.isfinite(swing_at_observed).all():
            Warn('The computation returned an infinite swing ratio. Returning for'
                 ' debugging purposes...', stacklevel=2)
            return (shift_party_seatshares, shift_party_voteshares, obs_party_voteshares)
        median_conints = est.intervals(party_seatshares, party_voteshares)
        observed_conints = est.intervals(shift_party_seatshares,
                                         shift_party_voteshares)
        swing_lm, swing_lm_resid = est.swing_slope(shift_party_seatshares,
                                                   shift_party_voteshares)

        self._swing_ratios_emp = swing_at_observed[0]
        self._swing_ratios_med = swing_near_median[0]
        self._swing_ratios_lm = swing_lm.mean()  # pool the parties in a 2party system
        self._swing_CIs = observed_conints
        self._swing_CIs_med = median_conints

        self._use_sim_swing = use_sim_swing

        return swing_at_observed[0] if use_sim_swing else swing_lm

    def _median_bonus_from_simulations(self, sims, q=[5, 50, 95], return_all=False):
        """
        Compute the bonus afforded to the reference party using:

        B = 2*E[s|v=.5] - 1

        where s is the seat share won by the reference party and v is the average vote share won by the reference party.
        """

        expected_seatshare = 2*(np.mean((sims > .5), axis=1)-.5)
        point_est = np.mean(expected_seatshare)
        point_est_std = np.std(expected_seatshare)
        if not return_all:
            return np.array([point_est - point_est_std*2,
                             point_est,
                             point_est + point_est_std*2])
        else:
            return expected_seatshare

    def _observed_bonus_from_simulations(self, sims,
                                         q=[5, 50, 95], return_all=False):
        """
        Compute the bonus afforded to the reference party by using:

        E[s | v=v_{obs}] - (1 - E[s | v = (1 - v_{obs})])

        where s is the seat share won by the reference party and v_{obs} is the observed share of the vote won by the reference party. This reduces to the difference in peformance between the reference party and the opponent when the opponent does as well as the reference.
        """
        raise NotImplementedError
        if year is not None:
            t = self.years.tolist().index(year)
        turnout, votes, observed_pvs, *rest = self._extract_election(t=t)
        observed_ref_share = observed_pvs[0]
        return self.winners_bonus_from_(n_sims=n_sims,
                                        target_v=observed_ref_share,
                                        t=t, Xhyp=Xhyp,
                                        predict=predict, q=q, return_all=return_all)

    def estimate_winners_bonus(self, n_sims=1000, t=-1, year=None,
                               target_v=.5, Xhyp=None, predict=False, q=[5, 50, 95], return_all=False):
        """
        Compute the bonus afforded to the reference party by using:

        E[s | v=v_i] - (1 - E[s | v = (1 - v_i)])

        where s is the seat share won by the reference party and v_i is an arbitrary target vote share. This reduces to a difference in performance between the reference party and the opponent when the opponent and the reference win `target_v` share of the vote.
        """
        raise NotImplementedError
        if year is not None:
            t = self.years.tolist().index(year)
        sims = self.simulate_elections(n_sims=n_sims, t=t, Xhyp=Xhyp, predict=predict,
                                       target_v=target_v, fix=True)
        complement = self.simulate_elections(n_sims=n_sims, t=t, Xhyp=Xhyp,
                                             predict=predict, target_v=1-target_v,
                                             fix=True)
        weights = 1/self.models[t].model.weights
        observed_expected_seats = np.mean(sims > .5, axis=1)  # what you won
        # what your oppo wins when you do as well as they did
        complement_opponent_seats = np.mean(1 - (complement > .5), axis=1)
        point_est = np.mean(observed_expected_seats -
                            complement_opponent_seats)
        point_est_std = np.std(observed_expected_seats -
                               complement_opponent_seats)
        if not return_all:
            return np.array([point_est - 2*point_est_std,
                             point_est,
                             point_est + 2*point_est_std])
        else:
            return observed_expected_seats - complement_opponent_seats

    def get_attainment_gap(self, t=-1, year=None, return_all=True):
        """
        Get the empirically-observed attainment gap, computed as the minimum vote share required to get a majority of the vote.

        G_a = ((.5 - s)/\hat{r} + v) - .5

        where s is the observed seat share, r is the estimated responsiveness in time t, and v is the party vote share in time t. Thus, the core of this statistic is a projection of the line with the responsiveness as the slope through the observed (v,s) point to a point (G_a,.5).

        Inherently unstable, this estimate is contingent on the estimate of the swing ratio.
        """
        raise NotImplementedError
        if not return_all:
            self._GIGO(
                'This cannot return all values, since it does not rely on simulation')
        if year is not None:
            t = list(self.years).index(year)
        try:
            return self._attainment_gap[t]
        except AttributeError:
            turnout, voteshare, *_ = self._extract_election(t)
            sr = self.get_swing_ratio(t=t)
            return est.attainment_gap(turnout, voteshare, sr)[0][0]

    def simulate_attainment_gap(self, t=-1, year=None, Xhyp=None, predict=False, q=[5, 50, 95],
                                n_sim_batches=1000, sim_batch_size=None,
                                best_target=None, return_all=False, **optimize_kws
                                ):
        """
        Estimate the attainment gap through simulation. Given a target vote share `best_target`,
        find the q'th quantiles (5,50,95 by default) of (.5 - minV) where minV is the smallest vote
        share in the batch (of size `sim_batch_size`) where the party stil retains a majority of the
        house. If this simulation is centered at the "optimal" attainment gap value from `optimal_attainment_gap`, 
        this should estimate percentile bounds on the smallest attainment gaps at that vote share. 

        For example, if best_target = .5, then this means `n_sim_batches` of simulations would be conducted
        where the average vote share over the entire batch was .5. Over these batches (each one of size `sim_batch_size`),
        all realizations where the party wins a majority are retained. Then, the minimum average vote share in these
        batches is computed and stored. 

        After all these minima are computed, the qth quantiles of these minima are returned. 
        They represent a the typical minimum vote share required by the party to win a majority. 
        `best_target`, then, simply represents a target for the search space. It should
        be small enough that the party occasionally wins very small majorities, but large enough that 
        they win at least one majority per `sim_batch_size`. 

        Arguments
        ----------
        t, year, Xhyp, predict (refer to self.simulate_elections)
        q       :   iterable
                    quantiles to use to summarize the minima
        n_sim_batches:  int
                        number of batches with which to simulate minima
        sim_batch_size: int
                        number of elections to simulate within each batch
        best_target:    float
                        vote share to center the batches
        **optimize_kws: keyword argument dictionary
                        passed to self.optimal_attainment_gap if no target 
                        is provided. 
        """
        raise NotImplementedError
        if year is None:
            year = self._years[t]
        elif year is not None:
            t = self._years.tolist().index(year)
        if sim_batch_size is None:
            sim_batch_size = n_sim_batches // 10
        if best_target is None:
            best_target = .5 + -1 * self.optimal_attainment_gap(t=t, year=year, Xhyp=Xhyp,
                                                                predict=predict, q=[
                                                                    50],
                                                                **optimize_kws)
        agaps = []
        weights = 1/self.models[t].model.weights
        counter = 0
        retry = 0
        for _ in tqdm(range(n_sim_batches),
                      desc='simulating with target={}'.format(best_target)):
            batch = self.simulate_elections(target_v=best_target, t=t, predict=predict,
                                            Xhyp=Xhyp, n_sims=sim_batch_size, fix=False)
            majorities = np.asarray(
                [((sim > .5).mean() > .5) for sim in batch])
            if not majorities.any():
                retry += 1
                continue
            candidate = np.average(
                batch[majorities], weights=weights, axis=1).min()
            agaps.append(candidate)
        if retry > 0:
            Warn('no majorities found in {} simulations! Configuration is: '
                 '\n\t target: \t{} '
                 '\n\t Xhyp is None: \t{}'
                 '\n\t batch_size: \t{}'
                 '\n\t n_batches: \t{}'
                 ''.format(retry, best_target, Xhyp is None,
                           sim_batch_size, n_sim_batches))
        if not return_all:
            return np.percentile(.5 - np.asarray(agaps), q=q)
        else:
            return .5 - agaps

    def optimal_attainment_gap(self, t=-1, year=None,
                               Xhyp=None, predict=False, q=[5, 50, 95],
                               n_batches=1000, batch_size=None,
                               loss='mad', return_all=False):
        """
        Returns the `q`th percentiles (5,50,95 by default) for (.5 - v*), where
        v* is the optimal statewide average vote share that minimizes a loss
        function:
        loss(.5 - E[s|v*])

        Where loss(.) may be mean absolute deviation or squared error loss. 

        In plain language, this is the excess statewide vote share (v* - .5) 
        that a party wins when it wins a *bare majority* (its share of seats is
        the smallest possible value above 50%) of the representative
        body. If this is negative, the party must typically win more than 
        50% of the votes to win 50% of the seats. If this is positive, 

        Arguments
        ---------
        t           : int
                      index of the time period to compute the attainment gap.
        year        : int
                      the year to compute the attainment gap. Supersedes `t` 
        Xhyp        : np.ndarray
                      a matrix of hypothetical electoral conditions under which 
                      to estimate the optimal attainment gap. 
        predict     : bool
                      whether to use the predictive form or counterfactual form
                      of the election simulators
        q           : iterable (tuple,list,array)
                      set of quantiles passed to numpy.quantile
        n_batches   : int
                      number of times to estimate the optimal attainment gap. Since
                      the gap is estimated many times over a stochastic objective,
                      this governs how many replications of the optimization problem
                      are conducted. 
        batch_size  : int
                      size of each simulation batch in the optimization problem. 
                      The total amount of simulated elections will be 
                      n_batches * (batch_size * nfev_per_batch), where nfev_per_batch
                      is the unknown number of times scipy.optimize.minimize_scalar
                      will evaluate the objective function. So, if this function is
                      very slow, batch_size is likely the critical path. 
        loss        : string
                      the option for loss function type, either 'mad', the mean
                      absolute deviation, or 'ssd', the sum of squared deviations.
                      If a callable, it must return a single scalar that represents
                      some distance metric about how far the seat shares in simulations
                      from the model in time t fall from having a bare majority.
        """
        raise NotImplemetnedError
        if year is None:
            year = self._years[t]
        if batch_size is None:
            batch_size = n_batches // 10
        elif year is not None:
            t = self._years.tolist().index(year)
        try:
            from scipy.optimize import minimize_scalar
        except ImportError:
            raise ImportError(
                'scipy.optimize is required to use this functionality')
        if isinstance(loss, str):
            if loss.lower() == 'mad':
                def seatgap(target):
                    """
                    the mean absolute gap between the observed seatshare and .5
                    """
                    sims = self.simulate_elections(target_v=target, t=t, n_sims=batch_size,
                                                   predict=predict, Xhyp=Xhyp)
                    seats = np.asarray([(sim > .5).mean()
                                        for sim in sims]).reshape(-1, 1)
                    mad = np.abs(seats - .5).mean()
                    return mad.item()
            elif loss.lower() == 'ssd':
                def seatgap(target):
                    """
                    The sum of squared deviations between the observed seatshare
                    and .5
                    """
                    sims = self.simulate_elections(target_v=target, t=t, n_sims=batch_size,
                                                   predict=predict, Xhyp=Xhyp)
                    seats = np.asarray([(sim > .5).mean()
                                        for sim in sims]).reshape(-1, 1)
                    ssd = (seats - .5).T.dot(seats - .5)
                    return ssd.item()
            else:
                raise KeyError('Form of seatgap loss function ({}) is not '
                               '("mad","ssd").'.format(loss.lower()))
        elif callable(loss):
            seatgap = loss
        else:
            raise TypeError('loss parameter not recognized as string ("mad", "ssd")'
                            ' or callable')
        best_targets = []
        for _ in tqdm(range(n_batches), desc='optimizing'):
            best_targets.append(minimize_scalar(seatgap,
                                                tol=1e-4,
                                                bounds=(.05, .95),
                                                method='bounded'))
        best_xs = np.asarray([op.x for op in best_targets if op.success])
        if not return_all:
            return np.percentile(.5 - best_xs, q=q)
        else:
            return .5 - best_xs

    def get_efficiency_gap(self, t=-1, year=None, voteshares=None, turnout=True, return_all=True):
        """
        Compute the percentage difference of wasted votes in a given election

        G_e = W_1 - W_2 / \sum_i^n m

        where W_k is the total wasted votes for party k, the number cast in excess of victory:

        W_k = sum_i^n (V_{ik} - m_i)_+

        Where V_{ik} is the raw vote cast in district i for party k and m_i is the total number of votes cast for all parties in district i
        """
        raise NotImplementedError
        if not return_all:
            self._GIGO(
                'This function has no ability to return all of its results, because it does not rely on simulations.')
        tvec, vshares, a, b, c = self._extract_election(t=t, year=year)
        vshares = voteshares if voteshares is not None else vshares
        if not isinstance(turnout, bool):
            return est.efficiency_gap(vshares[:, 0], turnout)
        elif turnout:
            return est.efficiency_gap(vshares[:, 0], tvec)
        else:
            return est.efficiency_gap(vshares[:, 0], turnout=None)

    def estimate_efficiency_gap(self, t=-1, year=None,
                                Xhyp=None, predict=False, n_sims=1000,
                                q=[5, 50, 95], turnout=True, return_all=False):
        """
        Compute the efficiency gap expectation over many simulated elections. 
        This uses the same estimator as `get_efficiency_gap`, 
        but computes the efficiency gap over many simulated elections.
        """
        raise NotImplementedError
        tvec, *rest = self._extract_election(t=t, year=year)
        if not isinstance(turnout, bool):
            tvec = turnout
        elif not turnout:
            tvec = None
        sims = self.simulate_elections(
            t=t,  Xhyp=Xhyp, predict=predict, n_sims=n_sims)
        gaps = [est.efficiency_gap(sim.reshape(-1, 1), turnout=tvec)
                for sim in sims]
        if not return_all:
            return np.percentile(gaps, q=q)
        else:
            return gaps

    def district_sensitivity(self, t=-1, Xhyp=None, predict=False, fix=False,
                             swing=None, n_sims=1000,
                             batch_size=None, n_batches=None,
                             reestimate=False, seed=2478879,
                             **jackknife_kw):
        """
        This computes the deletion simulations.
        t, Xhyp, predict, fix, swing, n_sims are all documented in simulate_elections.
        batch_size and n_batches refer to arguments to optimal_attainment_gap
        jackknife_kw refer to cvtools.jackknife arguments
        """
        raise NotImplementedError
        np.random.seed(seed)
        if n_batches is None:
            n_batches = n_sims
        if batch_size is None:
            batch_size = n_batches // 10
        original = copy.deepcopy(self.models[t])
        leverage = cvt.leverage(original)
        resid = np.asarray(original.resid).reshape(-1, 1)
        del_models = cvt.jackknife(original, full=True, **jackknife_kw)
        del_params = pd.DataFrame(np.vstack([d.params.reshape(1, -1) for d in del_models]),
                                  columns=original.params.index)

        if not reestimate:  # Then build the deleted models from copies of the original
            mods = (copy.deepcopy(original) for _ in range(int(original.nobs)))
            del_models = []
            for i, mod in enumerate(mods):
                mod.model.exog = np.delete(mod.model.exog, i, axis=0)
                mod.model.endog = np.delete(mod.model.endog, i)
                mod.model.weights = np.delete(mod.model.weights, i)
                del_models.append(mod)
        rstats = []

        # First, estimate full-map statistics
        full_mbon = self.estimate_median_bonus(t=t, Xhyp=Xhyp)
        full_obon = self.estimate_observed_bonus(t=t, Xhyp=Xhyp)
        full_egap_T = self.estimate_efficiency_gap(
            t=t, Xhyp=Xhyp, turnout=True)
        full_egap_noT = self.estimate_efficiency_gap(
            t=t, Xhyp=Xhyp, turnout=False)
        full_obs_egap_T = self.get_efficiency_gap(t=t, turnout=True)
        full_obs_egap_noT = self.get_efficiency_gap(t=t, turnout=False)
        full_agap = self.optimal_attainment_gap(t=t, Xhyp=Xhyp,
                                                batch_size=batch_size,
                                                n_batches=n_batches)

        # Then, iterate through the deleted models and compute
        # district sensivities in the target year (t).

        for idx, mod in tqdm(list(enumerate(del_models)), desc='jackknifing'):
            self.models[t] = mod
            del_vs = mod.model.endog[:, None]
            del_w = mod.model.weights
            del_X = mod.model.exog

            # make sure the hypothetical gets deleted as well
            del_Xhyp = np.delete(
                Xhyp, idx, axis=0) if Xhyp is not None else None

            # Compute various bias measures:
            # the observed efficiency gap (with/without turnout)
            obs_egap_t = est.efficiency_gap(del_vs, del_w)
            obs_egap_not = self.get_efficiency_gap(t=t, voteshares=del_vs,
                                                   turnout=False)
            # The median bonus
            mbon = self.estimate_median_bonus(t=t, Xhyp=del_Xhyp,
                                              n_sims=n_sims)
            # The observed bonus
            obon = self.estimate_observed_bonus(t=t, Xhyp=del_Xhyp,
                                                n_sims=n_sims)

            # The estimated (simulated) efficiency gap (with/without turnout)
            egap_T = self.estimate_efficiency_gap(t=t, Xhyp=del_Xhyp,
                                                  n_sims=n_sims,
                                                  turnout=mod.model.weights)
            egap_noT = self.estimate_efficiency_gap(t=t, Xhyp=del_Xhyp,
                                                    n_sims=n_sims,
                                                    turnout=False)
            agap = self.optimal_attainment_gap(t=t, Xhyp=del_Xhyp,
                                               n_batches=n_batches,
                                               batch_size=batch_size)
            rstats.append(np.hstack((obs_egap_t, obs_egap_not,
                                     mbon, obon, egap_T, egap_noT, agap)))
        # Reset the model for the time period back to the original model
        self.models[t] = original

        # prepare to ship everything by building columns & dataframe
        rstats = np.vstack(rstats)
        cols = (['EGap_eT', 'EGap_enoT']
                + ['{}_{}'.format(name, ptile)
                   for name in ['MBonus', 'OBonus', 'EGap_T', 'EGap_noT', 'AGap']
                   for ptile in (5, 50, 95)])
        rstats = pd.DataFrame(rstats, columns=cols)

        # and the leverage
        leverage = pd.DataFrame(np.hstack((np.diag(leverage).reshape(-1, 1),
                                           resid)),
                                columns=['leverage', 'residual'])
        dnames = self._designs[t].district_id

        # and the statewide estimates
        full_biases = pd.Series(np.hstack((full_obs_egap_T, full_obs_egap_noT,
                                           full_mbon, full_obon,
                                           full_egap_T, full_egap_noT, full_agap))).to_frame().T
        full_biases.columns = cols
        full_ests = pd.concat(
            (self.models[t].params.to_frame().T, full_biases), axis=1)
        full_ests['district_id'] = 'statewide'
        return pd.concat((full_ests,  # stack statewide on top of
                          pd.concat((dnames.reset_index(drop=True),  # district-level results
                                     del_params,
                                     leverage,
                                     rstats),
                                    axis=1, ignore_index=False)),
                         ignore_index=True, axis=0)

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


def _impute_unc(design, covariates, floor=.25, ceil=.75, fit_params=dict()):
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
    for yr, contest in design.groupby("year"):
        mask = (contest.vote_share < floor) | (contest.vote_share > (ceil))
        mask |= contest.vote_share.isnull()
        contested = contest[~mask]
        uncontested = contest[mask]
        unc_ix = uncontested.index
        imputor = sm.WLS(contested.vote_share,
                         sm.add_constant(
                             contested[covariates], has_constant='add'),
                         weights=contested.weight).fit(**fit_params)
        contest.ix[unc_ix, 'vote_share'] = imputor.predict(
            sm.add_constant(
                uncontested[covariates],
                has_constant='add'))
        imputed.append(contest)
    return pd.concat(imputed, axis=0)


def _impute_singlepass(design, covariates, floor=.01, ceil=.99, fit_params=dict()):
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
    wide = utils.make_designs(design,
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


def _impute_recursive(design, covariates, floor=.01, ceil=.99, fit_params=dict()):
    """
    This must iterate over each year, fit a model on that year and last 
    year (if available), and then predict that years' uncontesteds.
    Then, it must propagate these predictions forward. 
    """
    #design['vote_share__original'] = design['vote_share']
    # we're gonna fit models
    try:
        import statsmodels.api as sm
    except ImportError:
        Warn("Must have statsmodels installed to conduct imputation",
             category=ImportError, stacklevel=2)
        raise
    # use the same strategy of the uncontested variate as before
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
        contest = contest.merge(last_data[['district_id', 'vote_share']],
                                on='district_id', suffixes=('', '__prev'),
                                how='left')
        if contest.vote_share__prev.isnull().all():
            raise GIGOError('No match between two panels found in {}. Check that'
                            ' the district_id is correctly specified, in that it'
                            ' identifies districts uniquely within congresses, '
                            ' and can be used to join one year worth of data '
                            ' to another'.format(yr))
        if contest.redist.all():
            # if it's a redistricting cycle, impute like we don't have
            # the previous years' voteshares
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
        Xunc = sm.add_constant(uncontested[covariates], has_constant='add')
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
