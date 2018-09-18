import numpy as np
from warnings import warn
import pandas as pd

def make_designs(elex_frame, years=None, redistrict=None, district_id = 'district_id'):
    """
    Construct a sequence of pair-election design matrices from a long-format timeseries. 
    Let there be ni observations over T time period, for a total of N = \sum_t^T n_t 
    observations, where not all ni are equal. 

    Arguments
    ----------
    elex_frame: dataframe
                a dataframe containing the results of an election.
    years     : np.ndarray
                an N x 1 array containing the years of each record. 
    redistrict: np.ndarray
                an N x 1 array containing the years of each record.
    district_id: string
                 the name of the column in the dataframe to link geometries between congresses. 
                 Typically, this should be a "stateFIPS"+"district#", and redistricting 
                 information in `redistrict` governs when this is considered "continuous" 
                 vs. "discontinuous."
    """
    if years is None:
        years = elex_frame['year']
    if redistrict is None:
        Warn('computing redistricting from years vector')
        redistrict = census_redistricting(pd.Series(years))
    if district_id not in elex_frame.columns:
        raise KeyError("district_id provided ({}) not found in dataframe".format(district_id))

    working = elex_frame.copy()
    working['year'] = years
    working['redist'] = redistrict

    grouper = working.groupby('year')
    iterator = iter(grouper)

    out = []
    _, last_year = next(iterator)
    if district_id is None:
        last_year['district_id'] = np.arange(0, last_year.shape[0])
    else:
        last_year.rename(columns={district_id:'district_id'}, inplace=True)
    first_out = last_year.copy()
    out.append(first_out)
    for i, this_year in iterator:
        if district_id is None:
            this_year['district_id'] = np.arange(0,this_year.shape[0])
        else:
            this_year.rename(columns={district_id:'district_id'}, inplace=True)
        both_frames = pd.merge(this_year,
                               last_year[['district_id', 'vote_share']],
                               on='district_id', how='left',
                               suffixes=('', '__prev'))
        if (both_frames.redist == 1).all():
            both_frames.drop('vote_share__prev', axis=1, inplace=True)
        out.append(both_frames)
        last_year = this_year
    return out

def census_redistricting(years):
    return pd.Series(years).apply(lambda x: x % 10 == 2).apply(int)

def chol_mvn(Mu, Sigma):
    """
    Sample from a Multivariate Normal given a mean & Covariance matrix, using
    cholesky decomposition of the covariance. If the cholesky decomp fails due
    to the matrix not being strictly positive definite, then the
    numpy.random.multivariate_normal will be used.

    That is, new values are generated according to :
    New = Mu + cholesky(Sigma) . N(0,1)

    Parameters
    ----------
    Mu      :   np.ndarray (p,1)
                An array containing the means of the multivariate normal being
                sampled
    Sigma   :   np.ndarray (p,p)
                An array containing the covariance between the dimensions of the
                multivariate normal being sampled

    Returns
    -------
    np.ndarray of size (p,1) containing draws from the multivariate normal
    described by MVN(Mu, Sigma)
    """
    try:
        D = scla.cholesky(Sigma, overwrite_a = True)
        e = np.random.normal(0,1,size=Mu.shape)
        kernel = np.dot(D.T, e)
        out = Mu + kernel
    except np.linalg.LinAlgError:
        out = np.random.multivariate_normal(Mu.flatten(), Sigma)
        out = out.reshape(Mu.shape)
    return out

def check_psd(var, regularize=False):
    """
    This is a pretty dangerous function, and routine use should be avoided.

    It automatically regularizes a covariance matrix based on its smallest eigenvalue. Sometimes, martices that are nearly positive semi-definite have very small negative eigenvalues. Often, this is the smallest eigenvalue.

    So, this function picks the largest eigenvalue that is negative and adds it to the covariance diagonal. This regularizes the covariance matrix.
    """
    eigs = np.linalg.eigvals(var)
    if all(eigs >= 0):
        return True
    elif not regularize:
        raise ValueError('Covariance Matrix is not positive semi-definite, and'
                         ' has negative eigenvalues')
    elif regularize:
        first_negative = (eigs < 0).tolist().index(True)
        var += np.eye(var.shape[0]) * eigs[first_negative]
        Warn('Had to add {} to the covariance diagonal to make it PSD.'
             ' If this value is large, there is probably something substantively '
             ' wrong with your covariance matrix. '.format(eigs[first_negative]))
    return False, var

#############################
# Win Rules                 #
#############################

def plurality_wins(votes):
    """
    Computes the plurality winner of an election matrix, organized with columns as parties and rows as contests.

    Arguments
    ---------
    votes       :   np.ndarray
                    matrix of vote shares with (n,p) shape, where n is the number of contests and p the numbe of parties
    Returns
    ---------
    wins, a flat (n,) vector containing the column index of the winner in each row of `votes`.
    """
    wins = np.argmax(votes, axis=1)
    return wins

def majority_wins(votes):
    """
    Computes the majority winner of an election matrix, organized with columns as parties and rows as contests. If no candidate recieves a majority, the entry is assigned a missing value.

    Arguments
    ---------
    votes       :   np.ndarray
                    (n,p) array of n elections between p parties.

    Returns
    --------
    wins, the vector of victors that aligns with the columnspace of `votes`
    """
    which, wins = np.where(votes > .5)
    out = np.zeros((votes.shape[0],))
    out[which] = wins
    out[~which] = np.nan
    return out

#############################
# Working with Voting Data  #
#############################
