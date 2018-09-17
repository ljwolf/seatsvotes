from __future__ import division
import copy
import numpy as np
import pandas as pd
from warnings import warn as Warn
from . import utils as ut
from . import fit
from .. import cvtools as cvt
from ..mixins import Preprocessor, Plotter, AdvantageEstimator
from tqdm import tqdm


class Successive(Preprocessor, Plotter, AdvantageEstimator):
    def __init__(self, elex_frame, covariate_columns,
                 weight_column=None,
                 share_column='vote_share',
                 year_column='year', redistrict_column=None,
                 district_id='district_id',
                 missing='drop', uncontested=None):
        """
        Construct a Seats-Votes object for a given election. let ni contests in T time periods occur, 
        so that there are sum(ni)=N contests overall. 

        Arguments
        ---------
        elex_frame  :   pd.dataframe
                        dataframe (in long format) containing the data to be analyzed
        share_col   :   string
                        name of the column to use for the vote shares. 
        years       :   np.ndarray
                        N x 1 array of years in which the votes take place
        redistrict  :   np.ndarray
                        N x 1 array of indicator variables denoting when a redistricting occurs
        district_id :   string
                        name of column containing district ids
        missing     :   string
                        what to do when data is missing, either turnout, 
                        vote shares, or exogenous variables. 
        uncontested:   dictionary of configuration options. Keys may be:
                        method      :   method of resolving uncontested elections. Options include:
                                        drop - drop all uncontesteds from observations
                                        winsor - winsorize to the percentiles given in params
                                        censor - censor to the percentiles given in params
                                        shift - (default) do what JudgeIt does and move uncontesteds 
                                                to .25 or .75.
                        threshold   :   threshold past which elections are considered uncontested
                        variate     :   what to code the uncontested indicator as. May be:
                                        ordinal - (default) codes the variate as -1, 0, 1 denoting 
                                                  Republican uncontested, contested, Democrat uncontested
                                        categorical - codes the variate as three separate fixed effects. 
                        params      :   parameters to pass to the underlying resolution function. 
                                        For winsor, these are the percentiles to winsorize to. For
                                        censor, these are the percentiles to censor at. For shift, 
                                        these are the percentiles to move the uncontesteds to.
        """
        super().__init__(elex_frame,
                         covariates=covariate_columns,
                         weight_column=weight_column,
                         share_column=share_column,
                         year_column=year_column,
                         redistrict_column=redistrict_column,
                         district_id=district_id,
                         missing=missing, uncontested=uncontested)
        self._years = np.sort(self.long.year.unique())
        assert all([elex.year.unique() == year
                    for year, elex in zip(self.years, self.wide)]), "Years mismatch with designs in object.wide"
        self.models = fit.models(self._designs, share_col='vote_share',
                                 covariate_cols=self._covariate_cols,
                                 weight_col='weight')
        self._lambdas = [model.params.get('vote_share__prev', np.nan)
                         for model in self.models]
        self._lambda = np.nanmean(self._lambdas)
        #self._sigma2s = np.asarray([model.scale for model in self.models])
        self._sigma2s = self.get_modelattr('scale').values
        self._sigma2s[np.isinf(self._sigma2s)] = np.nan
        self._sigma2 = np.nanmean(self._sigma2s)

    ##############
    # Properties #
    ##############

    @property
    def _designs(self):
        return self.wide

    @property
    def p(self):
        return len(self._covariate_cols)

    @property
    def N(self):
        return self.long.shape[0]

    @property
    def Nt(self):
        return len(self.long)

    @property
    def years(self):
        return self._years

    @property
    def data(self):
        return pd.concat(self._designs, ignore_index=True)

    @property
    def params(self):
        """
        All of the parameters across all models
        """
        unite = pd.concat([model.params for model in self.models], axis=1)
        unite.columns = self.years
        return unite

    def get_modelattr(self, *attrs, years=None):
        """
        Create a dataframe of properties from the underlying models
        """
        candidate = pd.concat([pd.DataFrame([getattr(model, attr) for attr in attrs], columns=attrs)
                               for model in self.models],
                              axis=0)
        candidate.index = self.years.tolist()
        return candidate

    def simulate_elections(self, n_sims=10000, swing=None, Xhyp=None,
                           target_v=None, fix=False, t=-1, year=None,
                           predict=False):
        """
        Generic method to either compute predictive or counterfactual elections. Will always prefer computing counterfactuals to simulation.

        See also: predict, counterfactal

        Arguments
        ---------
        n_sims      :   int
                        number of simulations to conduct
        swing       :   float
                        arbitrary shift in vote means
        Xhyp        :   (n,k)
                        artificial data to use in the simulation
        target_v    :   float
                        target mean vote share to peg the simulations to. Will ensure that the average of all simulations conducted is this value.
        fix         :   bool
                        flag to denote whether each simulation is pegged exactly to `target_v`, or if it's only the average of all simulations pegged to this value.
        t           :   int
                        the target time offset to use for the counterfactual simulations. Overridden by year.
        year        :   int
                        the target year to use for the counterfactual simulations
        predict     :   bool
                        whether or not to use the predictive distribution or counterfactual distribution
        """
        t = self.years.tolist().index(year) if year is not None else t
        if swing is not None and target_v is not None:
            raise ValueError('either swing or target_v, not both.')
        if predict:
            sims = np.squeeze(self.predict(n_sims=n_sims, Xhyp=Xhyp))
        else:
            sims = np.squeeze(self.counterfactual(
                n_sims=n_sims,  t=t, Xhyp=Xhyp))
        # weight in prop to raw turnout, not variance weights
        turnout = (1/self.models[t].model.weights)
        if swing is not None:
            sims = sims + swing
        elif target_v is not None:
            # link the grand mean of simulations to the target vote value
            grand_mean = np.average(sims, weights=turnout, axis=1).mean()
            sims = sims + (target_v - grand_mean)
        if fix:
            # link each simulation mean to the target vote value
            sim_means = np.average(sims, weights=turnout,
                                   axis=1).reshape(-1, 1)
            if target_v is None:
                vt = self.models[t].model.endog
                target_v = np.average(vt, weights=turnout)
            if swing is not None:
                target_v += swing
            sims = sims + (target_v - sim_means)
        return np.clip(sims, 0, 1)

    def predict(self, n_sims=10000, Xhyp=None):
        """
        Generic method to either compute predictive or counterfactual elections.

        See also: predict, counterfactal

        Arguments
        ---------
        n_sims      :   int
                        number of simulations to conduct
        swing       :   float
                        arbitrary shift in vote means
        Xhyp        :   (n,k)
                        artificial data to use in the simulation
        target_v    :   float
                        target mean vote share to peg the simulations to. Will ensure that the average of all simulations conducted is this value.
        fix         :   bool
                        flag to denote whether each simulation is pegged exactly to `target_v`, or if it's only the average of all simulations pegged to this value.
        """
        if Xhyp is None:
            Xhyp = self.models[-1].model.exog
        sims = np.asarray([self._simulate_prediction(Xhyp)
                           for _ in range(n_sims)])
        return sims

    def _simulate_prediction(self, Xhyp):
        """
        Make a simulation from the estimated results
        """
        most_recent_betas = np.asarray(self.models[-1].params).reshape(-1, 1)
        mean = Xhyp.dot(most_recent_betas)
        # remember, these will be inverted
        weights = 1/self.models[-1].model.weights
        weights = np.diag(weights)
        covm = self.models[-1].cov_params()
        pred_var = weights * self._sigma2 + Xhyp.dot(covm).dot(Xhyp.T)
        vhyp = ut.chol_mvn(mean, pred_var)
        return vhyp

    def counterfactual(self, n_sims=10000, t=-1, Xhyp=None):
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
                        target mean vote share to peg the simulations to. Will ensure that the average of all simulations conducted is this value.
        fix         :   bool
                        flag to denote whether each simulation is pegged exactly to `target_v`, or if it's only the average of all simulations pegged to this value.
        """
        if t in self.years:
            year = t
            t = self.years.tolist().index(t)
        else:
            year = self.years[t]
        vt = self.models[t].model.endog.reshape(-1, 1)
        X = self.models[t].model.exog
        if Xhyp is None:
            Xhyp = X
        if Xhyp.shape != X.shape:
            raise Exception("Shape of counterfactual data does not match"
                            " existing data. {}".format(Xhyp.shape, X.shape))
        sims = np.asarray([self._simulate_counterfactual(year,  X, Xhyp, vt)
                           for _ in range(n_sims)])
        return sims

    def _simulate_counterfactual(self, year, X, Xhyp, known_v):
        """
        Draw directly from the posterior in Gelman and King 1994, equation 7.
        """
        t = self.years.tolist().index(year)
        betas = np.asarray(self.models[t].params).reshape(-1, 1)
        lam = self._lambda
        sig2 = self._sigma2
        shrinkv = lam * known_v
        shrinkX = Xhyp - lam * X
        mean = shrinkX.dot(betas) + shrinkv
        # again, they come inverted
        weights = np.diag(1/self.models[t].model.weights)
        system_vc = (1 - lam**2) * weights * sig2
        known_vcov = self.models[t].cov_params()
        cfact_vc = shrinkX.dot(known_vcov).dot(shrinkX.T)
        vhyp = ut.chol_mvn(mean, cfact_vc + system_vc)
        return vhyp

    def _draw_beta(self, t):
        """
        Implemented for the cascading simulation design in Gelman & King 1994,

        this is the sampling distribution of the MLE of beta.
        """
        means = np.asarray(self.models[t].params).reshape(-1, 1)
        vcov = self.models[t].cov_params()
        return ut.chol_mvn(means, vcov)

    def _draw_gamma(self, t, Xhyp, beta):
        """
        Implemented for the cascading simulation design in Gelman & King 1994

        This is equation 15 in Gelman & King 1994
        """
        lam = self._lambda
        sig2 = self._sigma2
        w = np.diag(1/self.models[t].model.weights)
        cov = lam * (1 - lam) * sig2 * w
        vt = self.models[t].model.endog.reshape(-1, 1)
        mean = lam*vt - lam*Xhyp.dot(beta)
        return ut.chol_mvn(mean, cov)

    def _draw_vhyp(self, t, Xhyp, beta, gamma):
        """
        Implemented for the cascading simulation design in Gelman & King 1994

        This is equation 14
        """
        lam = self._lambda
        sig2 = self._sigma2
        w = np.diag(1/self.models[t].model.weights)
        cov = (1 - lam) * sig2 * w
        mean = Xhyp.dot(beta) + gamma
        return ut.chol_mvn(mean, cov)

    def _draw_cf_conditionally(self, t, Xhyp=None):
        """
        Implemented for the cascading simulation design in Gelman & King 1994,
        following the directions on pg. 533
        """
        beta = self._draw_beta(t)
        gamma = self._draw_gamma(t, Xhyp, beta)
        vhyp = self._draw_vhyp(t, Xhyp, beta, gamma)
        return vhyp

    def _prefit(self, design,
                missing='drop', uncontested='judgeit',
                **uncontested_params):
        """
        This should
        1. remove cases where data is missing, both response and in the design
        2. resolve the uncontested elections where voteshares fall above or below a given threshold.
        """
        if missing.lower().startswith('drop'):
            design.dropna(
                subset=[['vote_share'] + self._covariate_cols], inplace=True)
        else:
            raise Exception('missing option {} not recognized'.format(missing))
        design = _unc_dispatch[uncontested.lower()](
            design, **uncontested_params)
        return design

###################################
# Dispatch Table for Uncontesteds #
###################################


def _censor_unc(design, lower=.25, upper=.75):
    """
    This will clip vote shares to the given mask.
    """
    design['vote_share'] = design.vote_share.clip(lower=lower, upper=upper)
    return design


def _shift_unc(design, lower=.05, upper=.95, lower_to=.25, upper_to=.75):
    """
    This replicates the "uncontested.default" method from JudgeIt, which replaces
    the uncontested elections (those outside of the (.05, .95) range) to (.25,.75).
    """
    lowers = design.query('vote_share < @lower').index
    uppers = design.query('vote_share > @upper').index
    design.ix[lowers, 'vote_share'] = lower_to
    design.ix[uppers, 'vote_share'] = upper_to
    return design


def _winsor_unc(design, lower=.25, upper=.75):
    """
    This winsorizes vote shares to a given percentile.
    """
    try:
        from scipy.stats.mstats import winsorize
    except ImportError:
        Warn('Cannot import scipy.stats.mstats.winsorize, censoring instead.',
             stacklevel=2)
        return _censor_unc(design, lower=lower, upper=1-upper)
    # WARNING: the winsorize function here is a little counterintuitive in that
    #          it requires the upper limit to be stated as "from the right,"
    #          so it should be less than .5, just like "lower"
    design['vote_share'] = np.asarray(winsorize(design.vote_share,
                                                limits=(lower, 1-upper)))
    return design


def _drop_unc(design, lower=.05, upper=.95):
    """
    This drops uncontested votes
    """
    mask = (design.vote_share > lower) * (design.vote_share < upper)
    return design[mask]


_unc_dispatch = dict(censor=_censor_unc,
                     shift=_shift_unc,
                     judgeit=_shift_unc,
                     winsor=_winsor_unc,
                     drop=_drop_unc)
