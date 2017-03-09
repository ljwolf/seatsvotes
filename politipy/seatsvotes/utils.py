from sklearn import mixture as mx
from sklearn.utils import check_random_state
import numpy as np
import warnings as warn
from scipy import linalg as scla

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
        D = scla.cholesky(Sigma, lower=True, overwrite_a = True)
        e = np.random.normal(0,1,size=Mu.shape)
        kernel = np.dot(D.T, e)
        out = Mu + kernel
    except np.linalg.LinAlgError:
        out = np.random.multivariate_normal(Mu.flatten(), Sigma)
        out = out.reshape(Mu.shape)
    return out


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
# Processing steps          #
#############################

def summarize_election(array, rule=plurality_wins):
    """
    Given an (k, n, 1+p) array containing of k realizations of an electoral system with n districts with turnout (first facet of the trailing dimension) and vote shares (all the rest in the trailing dimension) for p parties, compute:

    votes : (k,n,p) raw votes won by each party in each district in all simulations
    party_voteshares : (k,p) systemwide share of all votes cast won by each
                               party in each simultaion
    party_seats : (k,n) matrix denoting which party won each district in each simulation
    party_seatshares : (k,p) matrix containing the share of all seats won by the party in each simulation
    """
    if array.ndim == 2:
        array = array.reshape(1, *array.shape)
    turnout, shares = array[:,:,0:1], array[:,:,1:]
    k,N,p = shares.shape
    raw_votes = turnout * shares
    party_votes = raw_votes.sum(axis=1)
    party_voteshares = party_votes / party_votes.sum(axis=1).reshape(-1,1)
    party_seatwins = np.asarray([rule(a) for a in shares])
    party_vector = np.arange(0, p)
    countgen = (np.unique(counts, return_counts=True) for counts in party_seatwins)
    all_counts = []
    for hasparty, breakdown in countgen:
        has_parties = np.asarray([party in hasparty for party in party_vector])
        counts = np.zeros((p,))
        counts[has_parties] += breakdown
        all_counts.append(counts)
    party_seats = np.asarray(all_counts)
    party_seatshares = party_seats / party_seats.sum(axis=1).reshape(-1,1)
    return raw_votes, party_voteshares, party_seats, party_seatshares
