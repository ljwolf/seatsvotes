import numpy as np
from scipy import stats
from ..mixins import Preprocessor, AlwaysPredictPlotter


class Reswing(Preprocessor, AlwaysPredictPlotter):
    def __init__(self, elex_frame, covariate_columns=None,
                 weight_column=None,
                 share_column='vote_share',
                 year_column='year', redistrict_column=None, district_id='district_id',
                 missing='drop', uncontested=None):
        super().__init__(elex_frame,
                         covariates=covariate_columns,
                         weight_column=weight_column,
                         share_column=share_column,
                         year_column=year_column,
                         redistrict_column=redistrict_column,
                         district_id=district_id,
                         missing=missing,
                         uncontested=uncontested
                         )
        self._years = np.sort(self.long.year.unique())
        self._swing_mean = []
        self._swing_dev = []
        for i, elex in enumerate(self.wide):
            if 'vote_share__prev' in self.wide[i].columns:
                self.wide[i]['swing'] = elex.vote_share - elex.vote_share__prev
            else:
                self.wide[i]['swing'] = np.nan
            self._swing_mean += [self.wide[i]['swing'].mean()]
            self._swing_dev += [self.wide[i]['swing'].std()]

    @property
    def years(self):
        return self._years

    def simulate_elections(self, n_sims=1000,
                           t=-1, year=None, predict=True,
                           swing=0, target_v=None, fix=False):
        """
        Simulate elections according to a bootstrap technique. 

        Arguments
        ---------
        n_sims  :   int
                    number of simulations to conduct
        swing   :   float
                    arbitrary shift in vote means, will be added to the empirical distribution of $\delta_{t}$.
        target_v:   float
                    target mean vote share to peg the simulations to. Will ensure that the average of all 
                    simulations shift towards this value, but no guarantees about simulation expectation 
                    can be made due to the structure of the bootstrap. 
        t       :   int
                    the target time offset to use for the counterfactual simulations. Overridden by year.
        year    :   int
                    the target year to use for the counterfactual simulations
        predict :   bool
                    flag denoting whether to use the predictive distribution (i.e. add bootstrapped swings to 
                    the voteshare in the previous year) or the counterfactual distribution (i.e. add bootstrapped
                    swings to the voteshare in the current year).

        Returns
        ---------
        an (n_sims, n_districts) matrix of simulated vote shares. 
        """
        if fix:
            raise Exception(
                "Bootstrapped elections cannot be fixed in mean to the target value.")
        t = list(self.years).index(year) if year is not None else t
        turnout, _, party_voteshares, *rest = self._extract_election(year=year)
        this_year = self.wide[t]
        _swing = self.wide[t].swing
        if swing is not None and target_v is not None:
            raise ValueError("either swing or target_v, not both.")
        elif target_v is not None:
            swing = (target_v - party_voteshares[0])
        turnout, _, party_voteshares, *rest = self._extract_election(year=year)
        target_h = this_year.vote_share__prev if predict else this_year.vote_share
        n_dists = len(target_h)
        kde = stats.gaussian_kde(_swing)
        sim_swings = kde.resample(n_sims * n_dists).reshape(n_sims, n_dists)
        sim_h = np.clip(target_h.values[None, :] + sim_swings + swing, 0,1)
        return sim_h

    def _extract_election(self, *args, **kwargs):
        return self._extract_data(*args, **kwargs)
