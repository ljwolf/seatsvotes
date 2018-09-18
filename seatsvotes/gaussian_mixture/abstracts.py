from . import fit as _fit, preprocessing as prep
from ..mixins import AdvantageEstimator
from .. import utils as ut
from sklearn import mixture as mix
import numpy as np
import pandas as pd
import copy
from warnings import warn as Warn


class GaussianMixture(AdvantageEstimator):  # should inherit from preprocessor
    def __init__(self, elex_frame,
                 holdout=None, threshold=.95,
                 share_pattern='_share',
                 turnout_col='turnout',
                 year_col='year',
                 ):
        """
        Construct a Seats-Votes object for a given election.

        Arguments
        ---------
        data        :   dataframe
                        dataframe containing the elections to be analyzed
        holdout     :   string (default: first non-turnout column matching `share_pattern`)
                        party name to consider as the `holdout` party, against which
                        all log contrasts are constructed
        threshold   :   float (default: 95)
                        threshold beyond which all elections are considered uncontested.
        share_pattern:  string (default: `_share`)
                        pattern denoting how all the vote share column in the dataframe
                        are named. By default, all columns with `_share` in their name
                        are matched using dataframe.filter(like=share_pattern)
        turnout_col :   string
                        name of column containing turnout information

        """
        share_frame = elex_frame.filter(like=share_pattern)
        if share_frame.empty:
            raise KeyError("no columns in the input dataframe "
                           "were found that match pattern: {}".format(share_pattern))
        turnout = elex_frame.get(turnout_col)
        if threshold < .5:
            Warn('Threshold is an upper, not lower bound. Converting to upper bound.')
            threshold = 1 - threshold
        if holdout is None:
            holdout = elex_frame.columns[1]
        if isinstance(holdout, int):
            self._holdout_idx = holdout
        else:
            self._holdout_idx = list(elex_frame.columns).index(holdout)
        self._data = elex_frame
        self._share_cols = share_frame.columns.tolist()
        self._turnout_col = turnout_col
        self.turnout = turnout
        self.shares = share_frame
        self.N, self.P = share_frame.shape
        filtered = prep.filter_uncontested(self.shares, threshold)
        self.uncontested = elex_frame.drop(filtered.index, inplace=False)
        self.contested = elex_frame.drop(self.uncontested.index, inplace=False)
        self.n_uncontested = self.uncontested.shape[0]
        self._uncontested_threshold = threshold
        unc_d = (self.uncontested[self._share_cols].values >
                 self._uncontested_threshold).sum(axis=0)
        self._uncontested_p = unc_d / self.n_uncontested
        self.patterns = prep.extract_patterns(self.contested)
        contrasts = []
        hyperweights = []
        n_contested = 0
        for pattern in self.patterns:
            contrast = prep.make_log_contrasts(pattern.contests,
                                               holdout=holdout)
            contrasts.append(contrast)
            hyperweights.append(contrast.shape[0] / self.N)
            n_contested += contrast.shape[0]
        assert n_contested + self.n_uncontested == self.N, "missing/double-counting!"
        self.n_contested = n_contested
        self.contrasts = contrasts
        self._frac_contested = n_contested / self.N
        self.hyperweights = hyperweights
        self._max_size, self._argmax_size = np.max(self.P), np.argmax(self.P)
        self._has_been_fit = False

    def fit(self, n_components=None, ic='bic',
            model_kw=dict(), fit_kw=dict(), ic_kw=dict(),
            inplace=False):
        """
        Fit gaussian mixture models for a given collection of vote shares.

        Arguments
        ---------
        n_components    :   int or list of ints
                            number of components in each mixture model to fit. If only one integer is passed, all mixtures have the same number of components. If None, a search procedure optimizes the number of components according to `ic`.
        ic              :   str
                            the name of the information criteria to use if the number of components is unspecified. 'aic' and 'bic' are supported. See `GaussianMixture` for more.
        model_kw        :   dict of keyword arguments
                            arguments to pass to the `GaussianMixture()` call.
        fit_kw          :   dict of keyword arguments
                            arguments to pass to the `GaussianMixture().fit()` call
        ic_kw           :   dict of keyword arguments
                            arguments to pass to the information criterion call,
                            `GaussianMixture().fit().ic()`.
        """
        if n_components is None:
            n_components, _ = zip(*[_fit.optimize_degree(contrast)
                                    for contrast in self.contrasts])
        elif isinstance(n_components, int):
            n_components = [n_components] * len(self.contrasts)
        self.models = [mix.GaussianMixture(n_components=n_i, **model_kw)
                          .fit(contrast, **fit_kw)
                       for n_i, contrast in zip(n_components, self.contrasts)]
        self._has_been_fit = True
        if not inplace:
            return copy.deepcopy(self)

    def sample(self, n_samples, return_shares=True):
        """
        Construct a new set of `n_samples` vote shares (or log odds).

        Arguments
        ---------
        n_samples   :   int
                        number of samples to take from the composite density function, sampling of a mixture of mixtures
        shares      :   bool
                        flag governing whether to have the sampler return
                        vote shares shares or log odds
        """
        if not self._has_been_fit:
            raise Exception('Model must be fit first. Call `SeatsVotes.fit`')

        # Some will be uncontested.
        n_contested = np.random.binomial(n_samples, self._frac_contested)
        n_uncontested = n_samples - n_contested

        uncontesteds = self.draw_uncontesteds(n_uncontested)

        # Sample from mixture densities estimated over contests
        from_each = np.random.multinomial(n_contested, self.hyperweights)
        sample = []
        for i, from_this in enumerate(from_each):
            reference = np.zeros((from_this, self.P))
            candidate = _fit.fixed_sample(self.models[i], from_this)[0]
            candidate_votes = candidate[:, 1:]
            candidate_totals = candidate[:, 0].reshape(-1, 1)
            contrasts = np.hstack((np.zeros((from_this, 1)), candidate_votes))
            if return_shares:
                contrasts = np.exp(contrasts)
                shares = contrasts / contrasts.sum(axis=1).reshape(-1, 1)
            candidate_mask = np.asarray(self.patterns[i].mask)
            reference[:, candidate_mask] += shares
            output = np.hstack((candidate_totals, reference))
            sample.append(output)
        contesteds = np.vstack(sample)
        if uncontesteds is not None:
            out = np.vstack((uncontesteds, contesteds))
        else:
            out = contesteds
        return out

    def draw_uncontesteds(self, n_samples=1):
        """
        Draw some number of uncontested elections.

        Arguments
        ---------
        n_samples   :   int
                        number of uncontested elections to draw
        E
        """
        if not self._has_been_fit:
            raise Exception('Model must be fit first. Call `SeatsVotes.fit`')
        if n_samples == 0:
            return
        n_of_each = np.random.multinomial(n_samples, self._uncontested_p)
        turnout = []
        for i, ni in enumerate(n_of_each):
            colname = self.uncontested.columns[i+1]
            this_turnout = self.uncontested.query('{} >= {}'.format(colname,
                                                                    self._uncontested_threshold))\
                .turnout.sample(ni, replace=True).values
            turnout.extend(this_turnout)
        turnout = np.asarray(turnout).reshape(-1, 1)

        uncs = [[np.zeros(self.P,) + (k == n_of_each)]*k
                for k in n_of_each if k > 0]
        out = np.vstack(uncs).astype(int)
        out = np.hstack((turnout, out))
        return pd.DataFrame(out, columns=self._data.columns)

    def simulate_elections(self, n_sims=1000):
        """
        Construct a new set of `n_elections` elections.

        Arguments
        ---------
        n_elex      :   int
                        number of elections to simulate from the composite density function, sampling from a mixture of mixtures.

        Returns
        -------
        An array that is (n_elex, N,P)
        """
        if not self._has_been_fit:
            raise Exception('Model must be fit first. Call `SeatsVotes.fit`')
        out_df = (pd.DataFrame(self.sample(self.N),
                               columns=[self._turnout_col, *self._share_cols])
                  for _ in range(n_sims))
        out_df = (df.assign(run=i) for i, df in enumerate(out_df))
        out_df = pd.concat(out_df, axis=0)
        return out_df

    def compute_swing_ratio(self, n_sims=1000, rule=ut.plurality_wins,
                            percentiles=[5, 95], use_sim_swing=True):
        """
        Compute the swing ratio, or the slope of the seats-votes curve, in a small window around the observed points.
        """
        sim_elex = self.simulate_elections(n_sims)
        summary = ut.summarize_election(sim_elex, rule=rule)
        raw_votes, party_voteshares, party_seats, party_seatshares = summary
        observed_elex = np.hstack((self.turnout.values, self.shares.values))
        observed_summary = ut.summarize_election(observed_elex)
        obs_votes, obs_party_voteshares, obs_party_seats, obs_party_seatshares = observed_summary
        swing_emp = est.swing_about_pivot(party_seatshares, party_voteshares,
                                          obs_party_voteshares)
        conints = est.intervals(
            party_seatshares, party_voteshares, percentiles=percentiles)
        swing_lm, swing_lm_resid = est.swing_slope(
            party_seatshares, party_voteshares)

        self._swing_ratios_sim = swing_emp
        self._swing_ratios_lm = swing_lm
        self._swing_CIs = conints
        self._use_sim_swing = use_sim_swing

        return swing_emp if use_sim_swing else swing_lm

    @property
    def swing_ratios(self):
        if not hasattr(self, '_swing_ratios_sim'):
            self.compute_swing_ratio()
        if self._use_sim_swing:
            return self._swing_ratios_sim
        else:
            return self._swing_ratios_lm

    @property
    def swing_ratios_lm(self):
        if not hasattr(self, '_swing_ratios_lm'):
            self.compute_swing_ratio()
        return self._swing_ratios_lm

    @property
    def swing_CIs(self):
        if not hasattr(self._swing_intervals):
            self.compute_swing_ratio()
        return self._swing_CIs

    @property
    def winners_bonus(self):
        """
        The excess share of seats won by the party when it attains exactly 50% vote share, as suggested in King & Browning (1989). If negative, this indicates that the party would win fewer than 50% of the seats were they to win 50% of the votes.

        For example, if the winners bonus is .0221, then the party who has that bonus gains 52.21% of the seats in the legislature when it wins 50% of the vote.

        NOTE: this is pretty useless for multiparty systems.
        """
        if not hasattr(self, 'swing_ratios'):
            self.compute_swing_ratio()
        if not hasattr(self, '_winners_bonus'):
            self._winners_bonus, extrap = est.winners_bonus(self.turnout.values,
                                                            self.shares.values,
                                                            self.swing_ratios)
        return self._winners_bonus

    @property
    def pairwise_winners_bonus(self):
        """
        The the difference in winners bonuses for all parties when each party recieves a vote share of 50%.

        For example, if the pairwise winners bonus matrix is [[0,-.12],[0,0]], then party 1 wins 12% fewer seats than party 2 when they both win 50% of the votes. This is not necessarily the same as the winners bonus in two party systems, since the seats-votes curve can be asymmetric.
        """
        if not hasattr(self, 'swing_ratios'):
            self.compute_swing_ratio()
        if not hasattr(self, '_pairwise_winners_bonus'):
            self._pairwise_winners_bonus, _ = est.pairwise_winners_bonus(self.turnout.values,
                                                                         self.shares.values,
                                                                         self.swing_ratios)
        return self._pairwise_winners_bonus

    @property
    def attainment_bias(self):
        """
        The vote share below 50% that a party must win in order to win 50% of the seats in a legislature. If positive, the party must win more than 50% of the vote to win an outright majority in the legislature. This is the "mirror" or complement of the winners bonus.

        For example, if the attainment bias is -.091, then a party only needs to win 40.9% of the votes to win 50% of the seats in the legislature. If it were .023, the party would need to win 52.3% of the votes to win 50% of the seats in the legislature.
        """
        if not hasattr(self, 'swing_ratios'):
            self.compute_swing_ratio()
        if not hasattr(self, '_pairwise_winners_bonus'):
            self._attainment_bias, _ = est.attainment_bias(self.turnout.values,
                                                           self.shares.values,
                                                           self.swing_ratios)
        return self._attainment_bias

    @property
    def pairwise_attainment_bias(self):
        """
        The differences in vote shares required to win a majority of the legislature for all parties.

        For example, if the attainment bias matrix is [[0,-.12],[0,0]], then party 1 requires 12% fewer votes than party 2 in order to win a majority of the seats in the legislature. This is not necessarily the same as the attainment bias in two party systems, since the seats-votes curve can be asymmetric.
        """
        if not hasattr(self, 'swing_ratios'):
            self.compute_swing_ratio()
        if not hasattr(self, '_pairwise_attainment_bias'):
            self._pairwise_attainment_bias, _ = est.pairwise_attainment_bias(self.turnout.values,
                                                                             self.shares.values,
                                                                             self.swing_ratios)
        return self._pairwise_attainment_bias
