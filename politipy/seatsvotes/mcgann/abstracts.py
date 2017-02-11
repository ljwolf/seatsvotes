import numpy as np
import statsmodels.api as sm

def _year_to_decade(year):
    """
    A simple function so I don't mess this up later, this constructs the *redistricting*
    decade of a district. This is offset from the regular decade a year is in by two. 
    """
    return (yr - 2) - (yr - 2) % 10

class SeatsVotes(Preprocessor, Plotter):
    def __init__(self, frame,
                 share_column='vote_share',
                 covariates=None,
                 weight_column=None,
                 year_column='year',
                 redistrict_column=None,
                 district_id = 'district_id',
                 missing='drop',
                 uncontested=None,
                 break_on_GIGO=True):
        super().__init__(frame, share_column=share_column,
                         covariates=covariates,
                         weight_column=weight_column,
                         year_column=year_column,
                         redistrict_column=redistrict_column,
                         district_id=district_id,
                         missing=missing,
                         uncontested=uncontested,
                         break_on_GIGO=break_on_GIGO)
        self._decade_starts = np.sort(
                          list(
                          set([_year_to_decade(year)
                                for yr in self.years])))
        self.decades = {dec:[] for dec in self.decade_starts}
        for yr, wide in zip(self.years, self.wide):
            self.decades[_year_to_decade(yr)].append(wide)
        self.models = []
        for yr in self._decade_starts:
            self.decades[yr] = pd.concat(self.decades[yr], axis=0)
            self.models.append(sm.WLS(self.decades[yr].vote_share,
                                 sm.add_constant(self.decades[yr][self._covariate_cols]),
                                 weights=self.decades[yr].weights).fit())

    def simulate_elections(self, n_sims = 10000, t=-1, year=None, target_v=None, swing=0):
        if year is None:
            year = list(self.years)[t]
        else:
            t = list(self.years).index(year)
        decade = _year_to_decade(year)
        decade_t = list(self._decade_starts).index(decade)
        model = self.models[decade_t]
        X = self.wide[t][self._covariate_cols]
        expectation = model.predict(X)
        if target_v is not None:
            exp_pvs = np.average(expectation,weights=model.model.weights)
            diff = (target_v - exp_pvs)
            expectation += diff
        expectation += swing
        sims = np.random.normal(expectation, model.scale**.5, size=(n_sims, X.shape[0])) 
        return sims
