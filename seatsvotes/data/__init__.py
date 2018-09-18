import os
from collections import namedtuple
import numpy as np
import logging

self_dir = os.path.dirname(__file__)


def illinois():
    """ illinois elections to congress, including boundaries"""
    import geopandas as gpd
    return gpd.read_file('tar://' + os.path.join(self_dir, 'illinois.zip'))

def congress(state=None, geo=False, overwrite_cache=False):
    """Elections to the house from ICPSR 6311

    Arguments
    ----------
    state       :   string (default: None)
                    name of state to filter the data. all data is still
                    read into memory. 
    geo         :   bool   (default: False)
                    whether or not to use the geographies of the congressional
                    districts, provided by osf.io/vf9pf. Will be cached for 
                    future use if requested. 
    overwrite_cache:bool   (default: False)
                    if the spatial data is requested this will force the cached
                    version to be updated from the OSF if true. Otherwise,
                    the cached version will be read. 
    Returns
    -------
    (geo)dataframe containing election results, possibly with congressional
    district shapes (if and only if geo=True)
    """
    if not geo:
        return icpsr6311(state=state)
    else:
        _maybe_fetch_geodata_from_osf(overwrite_cache=overwrite_cache)
        return _load_geodata(state=state)
            
def _load_geodata(state=None):
    """ load the geodata from file, possibly filtering by a state"""
    try:
        import geopandas
    except ImportError:
        raise ImportError("package `geopandas` is required to use spatial data examples")
    path = os.path.join(self_dir, 'wolf_2018_scidata-osf-vf9pf.tar.gz')
    if not os.path.exists(path):
        path = _fetch_geodata_from_osf()
    full = geopandas.read_file('tar://' + path)
    if state is not None:
        return full.query('state_name == {}'.format(state.lower()))
    return full
    
def _maybe_fetch_geodata_from_osf(overwrite_cache=False):
    """ load the geodata from the open science framework, possibly overwriting
    the version cached on the hard drive
    """
    fh = os.path.join(self_dir, 'wolf_2018_scidata-osf-vf9pf.tar.gz')
    if os.path.exists(fh):
        if not overwrite_cache:
            return fh
    try:
        import osfclient
    except ImportError:
        raise ImportError("package `osfclient` is required to fetch spatial"
                          " congressional district data from the open "
                          " science framework")
    cxn = osfclient.OSF()
    wolf_scidata_storage = cxn.project('vf9pf').storage()
    files = list(wolf_scidata_storage.files)
    fnames = [f.name for f in files]
    file_ix = fnames.index("nineties_and_aughties.tar.gz")
    target_file = files[file_ix]
    logging.info('starting download from OSF.')
    with open(fh, 'wb') as f:
        target_file.write_to(f)
    logging.info('download from OSF successful.')
    return fh

def icspr6311(state=None):
    """ pull elections data from ICPSR6311, grabbed from judgeitII R package"""
    import pandas as pd
    house6311 = pd.read_csv(os.path.join(self_dir, 'judgeit/house6311.csv'))
    house6311.columns = ['year', 'idx', 'state',
                         'dist', 'inc', 'vote_share', 'turnout', 'south']
    if state is not None:
        return house6311.query('state == {}'.format(state.lower()))
    return house6311

def president(year=2012):
    """ Elections to the presidency by congressional district, from judgeitII R package"""
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
    """ Canadian parliamentary elections in 1979, from Linzer (2012)"""
    import pandas as pd
    out = pd.read_csv(os.path.join(self_dir, 'canada1979.csv'))
    out.columns = ['name', 'turnout', 'liberal',
                   'progcons', 'ndp', 'socialcredit']
    out.index = out.name
    return out
