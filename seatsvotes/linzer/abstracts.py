from . import fit as _fit, preprocessing as prep
from .. import estimators as est
from .. import utils as ut
from ..mixins import TwoPartyEstimator
from sklearn import mixture as mix
import numpy as np
import pandas as pd
import copy
from warnings import warn as Warn

class SeatsVotes(TwoPartyEstimator): # should inherit from preprocessor?
    def __init__(self, elex_frame, 
                 holdout=None, threshold=.95, 
                 share_pattern='_share',
                 turnout_col='turnout',
                 year_col='year',
                 **kws):
        """
        Construct a Seats-Votes object for a given election.

        Arguments
        ---------
        data        :   dataframe
        votecols    :   list of strings
        holdout     :   string
        kws         :   dict of keyword arguments
        """
        share_frame = elex_frame.filter(like=share_pattern)
        turnout = elex_frame.get(turnout_col)
        if threshold < .5:
            Warn('Threshold is an upper, not lower bound. Converting to upper bound.')
            threshold = 1 - threshold
        if holdout is None:
            holdout = share_frame.columns[1]
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
        self._twoparty = self.P <= 2
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
            n_components, _  = zip(*[_fit.optimize_degree(contrast)
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
        return_shares:  bool
                        flag governing whether to return the log odds contrasts
                        for parties, or to return vote shares directly.

        Returns
        -------
        (n_samples,p_parties) array, where p_parties is either 1, matching the 
        two-party omitted category pattern, or >2 for multiparty elections. 
        """
        if not self._has_been_fit:
            raise Exception('Model must be fit first. Call `SeatsVotes.fit`')

        ### Some will be uncontested.
        n_contested = np.random.binomial(n_samples, self._frac_contested)
        n_uncontested = n_samples - n_contested

        uncontesteds = self.draw_uncontesteds(n_uncontested)

        ### Sample from mixture densities estimated over contests
        from_each = np.random.multinomial(n_contested, self.hyperweights)
        sample = []
        for i, from_this in enumerate(from_each):
            reference = np.zeros((from_this, self.P))
            candidate = _fit.fixed_sample(self.models[i], from_this)[0]
            candidate_votes = candidate[:,1:]
            candidate_totals= candidate[:,0].reshape(-1,1)
            contrasts = np.hstack((np.zeros((from_this, 1)), candidate_votes))
            if return_shares:
                contrasts = np.exp(contrasts)
                shares = contrasts / contrasts.sum(axis=1).reshape(-1,1)
            candidate_mask = np.asarray(self.patterns[i].mask)
            reference[:,candidate_mask] += shares
            output = np.hstack((candidate_totals, reference))
            sample.append(output)
        contesteds = np.vstack(sample)
        if uncontesteds is not None:
            out = np.vstack((uncontesteds, contesteds))
        else:
            out = contesteds
        if self._twoparty:
            # state two-party races as omitted-category simulations, 
            # just like the rest of the library
            out = out[:,:-1]
            assert out.shape[1] == 2
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
        uncs = [[np.zeros(self.P,) + (k == n_of_each)]*k
                 for k in n_of_each if k > 0]
        out = np.vstack(uncs).astype(int)
        out = np.hstack((np.zeros((n_samples, 1)), out))
        return pd.DataFrame(out, columns=self._data.columns)

    def simulate_elections(self, n_sims = 1000, swing=0, 
                           target_v=None,
                           fix=False):
        """
        Construct a new set of `n_elections` elections.

        Arguments
        ---------
        n_elex      :   int
                        number of elections to simulate from the composite density function, sampling from a mixture of mixtures.

        Returns
        -------
        A long-form dataframe containing (n_observations * n_sims x 1+n_parties columns)
        or, two (n_sims x n_observations) arrays containing the turnout & vote shares
        in a two-party simulation
        """
        if ((swing != 0) or (target_v is not None)) and not self._twoparty:
            raise NotImplementedError("Multiparty swing not yet implemented.")
        if not self._has_been_fit:
            raise Exception('Model must be fit first. Call `SeatsVotes.fit`')
        out_df = (pd.DataFrame(self.sample(self.N))
                  for _ in range(n_sims))
        out_df = (df.assign(run = i) for i,df in enumerate(out_df))
        out_df = pd.concat(out_df, axis=0)
        if self._twoparty:
            if target_v is not None and swing != 0:
                raise ValueError("Provide only target_v or swing, not both.")
            if target_v is not None:
                swing = target_v - np.average(self.shares.iloc[:,0].values,
                                              self.turnout)
            turnouts = out_df.iloc[:,0].values.reshape(n_sims, self.N)
            votes = out_df.iloc[:,1].values.reshape(n_sims, self.N)
            return np.clip(votes + swing, 0,1), turnouts
        return out_df

    def _extract_election(self, *args, **kw):
        """ extract empirical elections from linzer model"""
        if self._twoparty:
            obs_vote_shares = self.share_frame.iloc[:,0].values
            obs_party_vote_shares = np.average(obs_vote_shares, 
                                               weight=self.turnout)
            obs_seats = (obs_vote_shares > .5).astype(int)
            obs_party_seat_shares = obs_seats.mean()
        else:
            obs_vote_shares = self.share_frame.values
            obs_party_vote_shares = np.average(obs_vote_shares, 
                                               weight=self.turnout,
                                               axis=0)
            wins = obs_vote_shares.argmax(axis=1)
            _, n_wins np.unique(wins, return_counts=True)
            obs_party_seat_shares = n_wins / n_wins.sum()

        return (np.asarray(self.turnout), 
                obs_vote_shares, obs_party_vote_shares,
                     wins, obs_party_seat_shares)