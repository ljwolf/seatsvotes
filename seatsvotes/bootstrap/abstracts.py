import numpy as np
from ..mixins import Preprocessor, AlwaysPredictPlotter, AdvantageEstimator
from warnings import warn


class Bootstrap(Preprocessor, AlwaysPredictPlotter, AdvantageEstimator):
    def __init__(self, elex_frame, covariate_columns=None,
                 weight_column=None,
                 share_column='vote_share',
                 year_column='year',
                 redistrict_column=None, district_id='district_id',
                 missing='ignore', uncontested='ignore'):
        super().__init__(elex_frame,
                         covariates=covariate_columns,
                         weight_column=weight_column,
                         share_column=share_column,
                         year_column=year_column,
                         redistrict_column=redistrict_column,
                         district_id=district_id,
                         missing=missing,
                         uncontested=uncontested,
                         )
        self._years = np.sort(self.long.year.unique())

    @property
    def years(self):
        return self._years

    def simulate_elections(self, n_sims=1000, predict=True,
                           t=-1, year=None, swing=0, target_v=None, fix=False, replace=True):
        """
        Simulate elections according to a bootstrap technique. 

        Arguments
        ---------
        n_sims  :   int
                    number of simulations to conduct
        swing   :   float
                    arbitrary shift in vote means, will be added to the 
                    empirical distribution of $\delta_{t}$.
        target_v:   float
                    target mean vote share to peg the simulations to. 
                    Will ensure that the average of all 
                    simulations shift towards this value, but no guarantees 
                    about simulation expectation 
                    can be made due to the structure of the bootstrap. 
        t       :   int
                    the target time offset to use for the counterfactual 
                    simulations. Overridden by year.
        year    :   int
                    the target year to use for the counterfactual simulations
        predict :   bool
                    flag denoting whether to use the predictive distribution 
                    (i.e. add bootstrapped swings to 
                    the voteshare in the previous year) or the counterfactual 
                    distribution (i.e. add bootstrapped
                    swings to the voteshare in the current year).
        fix     :   bool
                    flag denoting whether to force the average district vote to be
                    target_v exactly. If True, all elections will have exactly target_v
                    mean district vote share. If False, all elections will have approximately
                    target_v mean district vote share, with the grand mean vote share being target_v
        replace :   bool
                    flag denoting whether to resample swings with replacement or without replacement. 
                    If the sampling occurs without replacement, then each swing is used exactly one time in a simulation.
                    If the sampling occurs with replacement, then each swing can be used more than one
                    time in a simulation, and some swings may not be used in a simulation.
        Returns
        ---------
        an (n_sims, n_districts) matrix of simulated vote shares. 
        """
        if fix:
            raise Exception("Bootstrapped elections cannot be fixed in "
                            "mean to the target value.")
        t = list(self.years).index(year) if year is not None else t
        this_year = self.wide[t]
        party_voteshares = np.average(this_year.vote_share,
                                      weights=this_year.weight)
        if predict is False:
            self._GIGO("Prediction must be true if using bootstrap")
            target_h = this_year.vote_share.values.flatten()
        else:
            target_h = this_year.vote_share__prev.values.flatten()
        if swing is not None and target_v is not None:
            raise ValueError("either swing or target_v, not both.")
        elif target_v is not None:
            swing = (target_v - party_voteshares)
        obs_swings = (this_year.vote_share - this_year.vote_share__prev)
        obs_swings = obs_swings.fillna(obs_swings.mean())
        n_dists = len(target_h)
        pweights = (this_year.weight / this_year.weight.sum()).values.flatten()
        pweights /= pweights.sum()
        sim_swings = np.random.choice(obs_swings + swing, (n_sims, n_dists),
                                      replace=replace, p=pweights)
        sim_h = target_h[None, :] + sim_swings
        return np.clip(sim_h, 0, 1)
