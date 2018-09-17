import os
from collections import namedtuple
import numpy as np

self_dir = os.path.dirname(__file__)


def illinois():
    """ illinois elections to congress, including boundaries"""
    import geopandas as gpd
    return gpd.read_file('zip://' + os.path.join(self_dir, 'illinois.zip'))


def congress(state=None):
    """Elections to the house from ICPSR 6311"""
    import pandas as pd
    house6311 = pd.read_csv(os.path.join(self_dir, 'judgeit/house6311.csv'))
    house6311.columns = ['year', 'idx', 'state',
                         'dist', 'inc', 'vote_share', 'turnout', 'south']
    if state is not None:
        return house6311.query('state == {}'.format(state.lower()))
    return house6311


def president(year=2012):
    """ Elections to the presidency by congressional district, from JudgeIt"""
    import pandas as pd
    if isinstance(year, int):
        if year > 2012:
            raise Exception(
                'Only presidential elections since before 2012 are stored.')
        elif year in (2012, 2008):
            out = pd.read_csv(os.path.join(
                self_dir, 'pres_{}.csv'.format(int(year))))
        else:
            temp = pd.read_csv(os.path.join(self_dir, 'pres_upto_2004.csv'))
            out = temp.query('year == {}'.format(year))
            if out.empty:
                years = np.asarray([2012, 2008, *temp.year.unique()])
                close = years[np.argsort(np.abs(years - year))[0:2]]
                raise KeyError("Presidential year {} not found. Closest are:  {},{}"
                               .format(year, *close))
    elif isinstance(year, str):
        if year.lower() == 'all':
            p2012 = president(year=2012)
            p2008 = president(year=2008)
            rest = president(year='pre2008')
            oclass = namedtuple(
                'presidential', ['p2012', 'p2008', 'before_2008'])
            out = oclass(p2012, p2008, rest)
        elif year.lower() == 'pre2008':
            out = pd.read_csv(os.path.join(self_dir, 'pres_upto_2004.csv'))
    return out


def canada():
    """ Canadian parliamentary elections in 1979, from Linzer 2012"""
    import pandas as pd
    out = pd.read_csv(os.path.join(self_dir, 'canada1979.csv'))
    out.columns = ['name', 'turnout', 'liberal',
                   'progcons', 'ndp', 'socialcredit']
    out.index = out.name
    return out
