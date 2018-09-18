import numpy as np
import statsmodels.api as sm
import pandas as pd
from ..mixins import Preprocessor, AlwaysPredictPlotter, AdvantageEstimator


def _year_to_decade(yr):
    """
    A simple function so I don't mess this up later, this constructs the *redistricting*
    decade of a district. This is offset from the regular decade a year is in by two.
    """
    return (yr - 2) - (yr - 2) % 10


class Panel(Preprocessor, AlwaysPredictPlotter):
    def __init__(self, frame,
                 share_column='vote_share',
                 group_by='state',
                 covariate_columns=None,
                 weight_column=None,
                 year_column='year',
                 redistrict_column=None,
                 district_id='district_id',
                 missing='drop',
                 uncontested=None,
                 break_on_GIGO=True):
        super().__init__(frame, share_column=share_column,
                         covariates=covariate_columns,
                         weight_column=weight_column,
                         year_column=year_column,
                         redistrict_column=redistrict_column,
                         district_id=district_id,
                         missing=missing,
                         uncontested=uncontested,
                         break_on_GIGO=break_on_GIGO)
        self._years = np.sort(self.long.year.unique())
        self._covariate_cols += ['grouped_vs']
        self._decade_starts = np.sort(
            list(
                set([_year_to_decade(yr)
                     for yr in self.years])))
        self.decades = {dec: [] for dec in self._decade_starts}
        for i, (yr, wide) in enumerate(zip(self.years, self.wide)):
            if group_by is not None:
                grouped_vs = wide.groupby(
                    group_by).vote_share.mean().to_frame()
                grouped_vs.columns = ['grouped_vs']
                grouped_vs = grouped_vs.fillna(.5)
                self.wide[i] = wide.merge(
                    grouped_vs, left_on=group_by, right_index=True)
                self.wide[i]['grouped_vs'] = self.wide[i]['grouped_vs'].fillna(
                    .5)
            else:
                grouped_vs = wide.vote_share.mean()
                self.wide[i]['grouped_vs'] = grouped_vs
            self.decades[_year_to_decade(yr)].append(self.wide[i])
        self.models = []

        for yr in self._decade_starts:
            self.decades[yr] = pd.concat(self.decades[yr], axis=0, sort=True)

            # WLS Yields incredibly precise simulation values? Not sure why.
            X = sm.add_constant(self.decades[yr][self._covariate_cols]).values
            Y = self.decades[yr].vote_share.values
            Y[np.isnan(Y)] = self.decades[yr]['grouped_vs'].values[np.isnan(Y)]
            if weight_column is None:
                weights = None
                self.models.append(sm.GLS(Y, X).fit())
            else:
                weights = self.decades[yr].weight
                self.models.append(sm.GLS(Y, X, sigma=weights).fit())

    @property
    def years(self):
        return self._years

    @property
    def params(self):
        """
        All of the parameters across all models
        """
        unite = pd.concat([model.params for model in self.models], axis=1)
        unite.columns = self.years
        return unite

    def simulate_elections(self, n_sims=1000, t=-1, year=None,
                           target_v=None, swing=0., fix=False, predict=True):
        if year is None:
            year = list(self.years)[t]
        else:
            t = list(self.years).index(year)
        decade = _year_to_decade(year)
        decade_t = list(self._decade_starts).index(decade)
        model = self.models[decade_t]
        mask = (self.decades[decade].year == year)
        X = np.asarray(self.wide[t][self._covariate_cols])
        expectation = model.predict(sm.add_constant(X, has_constant='add'))
        if target_v is not None:
            exp_pvs = np.average(expectation, weights=self.wide[t].weight)
            diff = (target_v - exp_pvs)
            expectation += diff
        if swing is not None:
            expectation += swing
        # grab the square of the cov relating to the simulations and cast to std. dev.
        sigma = model.model.sigma[mask]**.5
        sigma *= model.scale ** .5
        sims = np.random.normal(expectation, sigma, size=(n_sims, X.shape[0]))
        if fix:
            sims -= sims.mean(axis=0)
        return sims
