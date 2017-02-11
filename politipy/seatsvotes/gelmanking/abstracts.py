from __future__ import division
import copy
import numpy as np
import pandas as pd
from warnings import warn as Warn
from scipy.stats import rankdata
from collections import OrderedDict
from . import utils as ut
from . import fit
from .. import estimators as est
from .. import cvtools as cvt
from ..mixins import Preprocessor, Plotter

class SeatsVotes(Preprocessor, Plotter):
    def __init__(self, elex_frame, covariate_columns,
                 weight_column=None,
                 share_column='vote_share',
                 year_column='year', redistrict_column=None, district_id='district_id',
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
                        what to do when vote shares are missing.
        uncontesteds:   dictionary of configuration options. Keys may be:
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
                              covariates = covariate_columns, 
                              weight_column=weight_column, 
                              share_column=share_column,
                              year_column=year_column,
                              redistrict_column = redistrict_column, 
                              district_id=district_id,
                              missing=missing, uncontested=uncontested)
        self._years = np.sort(self.long.year.unique())
        assert all([elex.year.unique() == year 
                    for year,elex in zip(self.years, self.wide)]), "Years mismatch with designs in object.wide"
        self.models = fit.models(self._designs, share_col='vote_share',
                                 covariate_cols = self._covariate_cols,
                                 weight_col ='weight')
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
            sims = np.squeeze(self.counterfactual(n_sims=n_sims,  t=t, Xhyp=Xhyp))
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
            sim_means = np.average(sims, weights=turnout, axis=1).reshape(-1,1)
            if target_v is None:
                vt = self.models[t].model.endog
                target_v = np.average(vt, weights=turnout)
            if swing is not None:
                target_v += swing
            sims = sims + (target_v - sim_means)
        return sims

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
        vt_last = self.models[-1].model.endog.mean()
        if Xhyp is None:
            Xhyp = self.models[-1].model.exog
        sims = np.asarray([self._simulate_prediction(Xhyp)
                            for _ in range(n_sims)])
        return sims

    def _simulate_prediction(self, Xhyp):
        most_recent_betas = np.asarray(self.models[-1].params).reshape(-1,1)
        mean = Xhyp.dot(most_recent_betas)
        weights = 1/self.models[-1].model.weights #remember, these will be inverted
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
        vt = self.models[t].model.endog.reshape(-1,1)
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
        betas = np.asarray(self.models[t].params).reshape(-1,1)
        lam = self._lambda
        sig2 = self._sigma2
        shrinkv = lam * known_v
        shrinkX = Xhyp - lam * X
        mean = shrinkX.dot(betas) + shrinkv
        weights = np.diag(1/self.models[t].model.weights) # again, they come inverted
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
        means = np.asarray(self.models[t].params).reshape(-1,1)
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
        vt = self.models[t].model.endog.reshape(-1,1)
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
        cov  = (1 - lam) * sig2 * w
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
                        target mean vote share to peg the simulations to. Will ensure that the average of all simulations conducted is this value.
        fix         :   bool
                        flag to denote whether each simulation is pegged exactly to `target_v`, or if it's only the average of all simulations pegged to this value.
        predict     :   bool
                        whether or not to use the predictive distribution or the counterfactual distribution
        use_sim_swing:  bool
                        whether to use the instantaneous change observed in simulations around the observed seatshare/voteshare point, or to use the aggregate slope of the seats-votes curve over all simulations as the swing ratio
        """
        ### Simulated elections
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

        ## Swing Around Median
        party_voteshares = np.hstack((ref_voteshares.reshape(-1,1),
                                      1-ref_voteshares.reshape(-1,1)))
        party_seatshares = np.hstack((ref_seatshares.reshape(-1,1),
                                      1-ref_seatshares.reshape(-1,1)))

        swing_near_median = est.swing_about_pivot(party_seatshares,
                                                  party_voteshares,
                                            np.ones_like(obs_party_voteshares)*.5)

        ## Swing near observed voteshare
        shift_simulations = simulations + (obs_party_voteshares[0] - .5)
        shift_ref_voteshares = np.average(shift_simulations,
                                          weights=turnout, axis=1)
        shift_ref_seatshares = (shift_simulations > .5).mean(axis=1)

        shift_party_voteshares = np.hstack((shift_ref_voteshares.reshape(-1,1),
                                      1-shift_ref_voteshares.reshape(-1,1)))
        shift_party_seatshares = np.hstack((shift_ref_seatshares.reshape(-1,1),
                                      1-shift_ref_seatshares.reshape(-1,1)))

        swing_at_observed = est.swing_about_pivot(shift_party_seatshares,
                                                  shift_party_voteshares,
                                                  obs_party_voteshares)
        ## Sanity Check
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
        self._swing_ratios_lm = swing_lm.mean() #pool the parties in a 2party system
        self._swing_CIs = observed_conints
        self._swing_CIs_med = median_conints

        self._use_sim_swing = use_sim_swing

        return swing_at_observed[0] if use_sim_swing else swing_lm

    @property
    def swing_ratios(self):
        """
        All swing ratios for all time periods. May take a while to compute.
        """
        if not hasattr(self, '_swing_ratios_sim_all'):
            emps, meds, lms, empCI, medCI = [], [], [], [], []
            for t, _ in enumerate(self.models):
                self.get_swing_ratio(t=t, predict=False)
                emps.append(copy.deepcopy(self._swing_ratios_emp))
                meds.append(copy.deepcopy(self._swing_ratios_med))
                lms.append(copy.deepcopy(self._swing_ratios_lm))
                empCI.append(self._swing_CIs)
                medCI.append(self._swing_CIs_med)
            self._swing_ratios_sim_all = np.asarray(emps)
            self._swing_ratios_lm_all = np.asarray(lms)
            self._swing_ratios_med_all = np.asarray(meds)
            self._swing_CIs_all = np.asarray(empCI)
            self._swing_CIs_med_all = np.asarray(medCI)
        if self._use_sim_swing:
            return self._swing_ratios_sim_all
        else:
            return self._swing_ratios_lm_all

    @property
    def swing_ratios_lm(self):
        """
        All swing ratios, computed using linear regression on simulated elections.
        """
        if not hasattr(self, '_swing_ratios_lm'):
            _ = self.swing_ratios
        return self._swing_ratios_lm_all

    @property
    def swing_CIs(self):
        """
        All confidence intervals of the Seats/Votes curve.
        """
        if not hasattr(self._swing_intervals):
            _ = self.swing_ratios
        return self._swing_CIs_all

    def estimate_median_bonus(self, n_sims=1000, t=-1, year = None,
                              Xhyp=None, predict=False, q=[5,50,95]):
        """
        Compute the bonus afforded to the reference party using:

        B = 2*E[s|v=.5] - 1

        where s is the seat share won by the reference party and v is the average vote share won by the reference party.
        """
        if year is not None:
            t = self.years.tolist().index(year)
        sims = self.simulate_elections(n_sims=n_sims, t=t, Xhyp=Xhyp, predict=predict,
                                       target_v=.5, fix=True)
        weights = 1/self.models[t].model.weights
        median_seatshare = np.average((sims>.5), weights=weights, axis=1)
        return np.percentile((median_seatshare*2) -1, q=q)

    def estimate_observed_bonus(self, n_sims=1000, t=-1, year = None,
                                Xhyp=None, predict=False, q=[5,50,95]):
        """
        Compute the bonus afforded to the reference party by using:

        E[s | v=v_{obs}] - (1 - E[s | v = (1 - v_{obs})])

        where s is the seat share won by the reference party and v_{obs} is the observed share of the vote won by the reference party. This reduces to the difference in peformance between the reference party and the opponent when the opponent does as well as the reference.
        """
        if year is not None:
            t = self.years.tolist().index(year)
        turnout, votes, observed_pvs, *rest = self._extract_election(t=t)
        observed_ref_share = observed_pvs[0]
        return self.estimate_winners_bonus(n_sims=n_sims, 
                                           target_v = observed_ref_share,
                                           t=t, Xhyp=Xhyp, predict=predict, q=q)

    def estimate_winners_bonus(self, n_sims=1000, t=-1, year = None,
                               target_v=.5, Xhyp=None, predict=False, q=[5,50,95]):
        """
        Compute the bonus afforded to the reference party by using:

        E[s | v=v_i] - (1 - E[s | v = (1 - v_i)])

        where s is the seat share won by the reference party and v_i is an arbitrary target vote share. This reduces to a difference in performance between the reference party and the opponent when the opponent and the reference win `target_v` share of the vote.
        """
        if year is not None:
            t = self.years.tolist().index(year)
        sims = self.simulate_elections(n_sims=n_sims, t=t, Xhyp=Xhyp, predict=predict,
                                   target_v = target_v, fix=True)
        complement = self.simulate_elections(n_sims=n_sims, t=t, Xhyp=Xhyp,
                                             predict=predict, target_v=1-target_v,
                                             fix=True)
        weights = 1/self.models[t].model.weights
        observed_expected_seats = np.mean(sims>.5, axis=1)
        complement_expected_seats = np.mean(complement>.5, axis=1)
        return np.percentile(observed_expected_seats - (1-complement_expected_seats),
                           q=q)

    def get_attainment_gap(self, t=-1, year=None):
        """
        Get the empirically-observed attainment gap, computed as the minimum vote share required to get a majority of the vote.

        G_a = ((.5 - s)/\hat{r} + v) - .5

        where s is the observed seat share, r is the estimated responsiveness in time t, and v is the party vote share in time t. Thus, the core of this statistic is a projection of the line with the responsiveness as the slope through the observed (v,s) point to a point (G_a,.5).

        Inherently unstable, this estimate is contingent on the estimate of the swing ratio.
        """
        if year is not None:
            t = list(self.years).index(year)
        try:
            return self._attainment_gap[t]
        except AttributeError:
            turnout, voteshare, *_ = self._extract_election(t)
            sr = self.get_swing_ratio(t=t)
            return est.attainment_gap(turnout, voteshare, sr)[0][0]

    def estimate_attainment_gap(self, t=-1, year=None, Xhyp=None, predict=False, q=[5,50,95],
                                 n_batches=100, batch_size=100, best_target=None):
        """
        Estimate the attainment gap through simulation. This computes

        G_a = (\sum_i^b \min_j^n({v_{ij} | s_{ij} > .5}) / b) - .5

        Where b is a number of simulation batches, n is the number of simulations-per-batch, 
        v_{ij} is the reference paty vote share in batch j for replication i, 
        s_{ij} is the reference party seat share in batch j for replication i.
        """
        if best_target is None:
            try:
                from scipy.optimize import brent
            except ImportError:
                raise ImportError('scipy.optimize is required to use this functionality')
            observed_egap = self.get_attainment_gap(t=t)
            def seatgap(target):
                sims = self.simulate_elections(target_v=target, t=t, n_sims=100,
                                               predict=predict, Xhyp=Xhyp)
                seats = np.asarray([(sim > .5).mean() for sim in sims]).reshape(-1,1)
                ssq = (seats - .5).T.dot(seats - .5)
                return ssq
            best_target = brent(seatgap, tol=1e-2, brack=(.05,.95))
        agaps = []
        for _ in range(n_batches):
            batch = self.simulate_elections(target_v=best_target, t=t, predict=predict,
                                            Xhyp=Xhyp, n_sims=batch_size)
            majorities = np.asarray([((sim > .5).mean() > .5) for sim in batch])
            candidate = batch[majorities].mean(axis=1).min()
            agaps.append(candidate - .5)
        return np.percentile(agaps, q=q)

    def get_efficiency_gap(self, t=-1, year=None, use_turnout=True):
        """
        Compute the percentage difference of wasted votes in a given election

        G_e = W_1 - W_2 / \sum_i^n m

        where W_k is the total wasted votes for party k, the number cast in excess of victory:

        W_k = sum_i^n (V_{ik} - m_i)_+

        Where V_{ik} is the raw vote cast in district i for party k and m_i is the total number of votes cast for all parties in district i
        """
        turnout, voteshares, a, b, c = self._extract_election(t=t,year=year)
        if use_turnout:
            return est.efficiency_gap(voteshares[:,0], 
                                      turnout)
        else:
            return est.efficiency_gap(voteshares[:,0], turnout=None)

    def estimate_efficiency_gap(self, t=-1, year=None, 
                                Xhyp=None, predict=False, n_sims=1000,
                                q=[5,50,95], use_turnout=True):
        """
        Compute the efficiency gap expectation over many simulated elections. This uses the same estimator as `get_efficiency_gap`, but computes the efficiency gap over many simulated elections.
        """
        turnout, *rest = self._extract_election(t=t, year=year)
        if not use_turnout:
            turnout = None
        sims = self.simulate_elections(t=t,  Xhyp=Xhyp, predict=predict, n_sims=n_sims)
        gaps = [est.efficiency_gap(sim.reshape(-1,1), turnout=turnout)
                for sim in sims]
        return np.percentile(gaps, q=q)
    
    def district_sensitivity(self, t=-1, Xhyp=None, predict=False, fix=False,
                             swing=None, n_sims=1000, reestimate=False,
                             **jackknife_kw):
        """
        This computes the deletion simulations.
        """
        original = copy.deepcopy(self.models[t])
        leverage = cvt.leverage(original)
        resid = np.asarray(original.resid).reshape(-1,1)
        del_models = cvt.jackknife(original, full=True, **jackknife_kw)
        del_params = pd.DataFrame(np.vstack([d.params.reshape(1,-1) for d in del_models]),
                                  columns=original.params.index)

        if not reestimate: #Then build the deleted models from copies of the original
            mods = (copy.deepcopy(original) for _ in range(int(original.nobs)))
            del_models = []
            for i,mod in enumerate(mods):
                mod.model.exog = np.delete(mod.model.exog, i, axis=0)
                mod.model.endog = np.delete(mod.model.endog, i)
                mod.model.weights = np.delete(mod.model.weights, i)
                del_models.append(mod)
        rstats = []

        # First, estimate full-map statistics
        full_mbon = self.estimate_median_bonus(t=t, Xhyp=Xhyp)
        full_obon = self.estimate_observed_bonus(t=t, Xhyp=Xhyp)
        full_egap_T = self.estimate_efficiency_gap(t=t, Xhyp=Xhyp, use_turnout=True)
        full_egap_noT = self.estimate_efficiency_gap(t=t, Xhyp=Xhyp, use_turnout=False)
        full_obs_egap_T = self.get_efficiency_gap(t=t, use_turnout=True)
        full_obs_egap_noT = self.get_efficiency_gap(t=t, use_turnout=False)

        # Then, iterate through the deleted models and compute 
        # district sensivities in the target year (t). 
        for idx,mod in enumerate(del_models):
            self.models[t] = mod
            # make sure the hypothetical gets deleted as well
            Xhyp_i = np.delete(Xhyp, idx, axis=0) if Xhyp is not None else None
            
            # Compute various bias measures:
            # the observed efficiency gap (with/without turnout)
            obs_egap_t = self.get_efficiency_gap(t=t)
            obs_egap_not = self.get_efficiency_gap(t=t, use_turnout=False)

            # The median bonus
            mbon = self.estimate_median_bonus(t=t, Xhyp=Xhyp_i, 
                                              n_sims=n_sims)
            # The observed bonus
            obon = self.estimate_observed_bonus(t=t, Xhyp=Xhyp_i, 
                                                n_sims=n_sims)

            # The estimated (simulated) efficiency gap (with/without turnout)
            egap_T = self.estimate_efficiency_gap(t=t, Xhyp=Xhyp_i, 
                                                  n_sims=n_sims, 
                                                  use_turnout=True)
            egap_noT = self.estimate_efficiency_gap(t=t, Xhyp=Xhyp_i, 
                                                    n_sims=n_sims,
                                                    use_turnout=False)
            rstats.append(np.hstack((obs_egap_t, obs_egap_not, 
                                     mbon, obon, egap_T, egap_noT)))
        # Reset the model for the time period back to the original model
        self.models[t] = original

        # prepare to ship everything by building columns & dataframe
        rstats = np.vstack(rstats)
        cols = ( ['EGap_eT', 'EGap_enoT']
                + ['{}_{}'.format(name,ptile) 
                    for name in ['MBonus', 'OBonus', 'EGap_T', 'EGap_noT']
                    for ptile in (5,50,95)] )
        rstats = pd.DataFrame(rstats, columns=cols)

        # and the leverage
        leverage = pd.DataFrame(np.hstack((np.diag(leverage).reshape(-1,1), 
                                           resid)),           
                                columns=['leverage', 'residual'])
        dnames = self._designs[t].district_id

        # and the statewide estimates
        full_biases = pd.Series(np.hstack((full_obs_egap_T, full_obs_egap_noT,
                                           full_mbon, full_obon, 
                                           full_egap_T, full_egap_noT))).to_frame().T
        full_biases.columns = cols
        full_ests = pd.concat((self.models[t].params.to_frame().T, full_biases), axis=1)
        full_ests['district_id'] = 'statewide'
        return pd.concat((full_ests, # stack statewide on top of  
                          pd.concat((dnames, del_params, #district-levels
                                     leverage, rstats), axis=1)))

    def _prefit(self, design,
                missing='drop', uncontested='judgeit',
                **uncontested_params):
        """
        This should
        1. remove cases where data is missing, both response and in the design
        2. resolve the uncontested elections where voteshares fall above or below a given threshold.
        """
        if missing.lower().startswith('drop'):
            design.dropna(subset=[['vote_share'] + self._covariate_cols], inplace=True)
        else:
            raise Exception('missing option {} not recognized'.format(missing))
        design = _unc_dispatch[uncontested.lower()](design, **uncontested_params)
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
        return _censor_unc(shares, lower=lower, upper=1-upper)
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
    mask = (design.vote_share < outer) * (design.vote_share > (1 - outer))
    return design[mask]

_unc_dispatch = dict(censor=_censor_unc,
                     shift=_shift_unc,
                     judgeit=_shift_unc,
                     winsor=_winsor_unc,
                     drop=_drop_unc)
