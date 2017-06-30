import numpy as np
import pandas as pd
from . import utils as ut

def swing_about_pivot(party_seat_shares, party_vote_shares,
                          observed_pvs, window=.02):
    """
    Computes the swing ratio as the instantaneous change in the seats a party wins for small changes in votes.

    Arguments
    ----------
    party_seat_shares   :   np.ndarray (n,p)
                            array containing the fraction of overall seats party j wins, j = 1, 2, ... , p over n replications
    party_vote_shares   :   np.ndarray (n,p)
                            array containing the faction of overall votes party j wins, j = 1, 2, ..., p over n replications
    observed_pvs        :   np.ndarray (1,p)
                            array containing the fraction of votes a party did win. This can be a vector of the observed average party vote share, or a targeted synthetic vote breakdown.
    window              :   float
                            the size of the window around the observed vote to search. Will be extended by 10% to capture rounding errors correctly

    Returns
    -------
    swing_emp (p,) the vector of average change in the window around observed_pvs
    """
    observed_pvs = np.asarray(observed_pvs)
    hi = (window*.5) + .1*window
    lo = (window*.5) - .1*window
    near_high = (party_vote_shares < (observed_pvs + hi))
    near_high &= (party_vote_shares > (observed_pvs + lo))
    near_low = (party_vote_shares > (observed_pvs - hi))
    near_low &= party_vote_shares < (observed_pvs - lo)
    # need the column average of seat shares in within the window
    # so the sum of the filtered column, divided by the size of each filter column
    swing_hi = (party_seat_shares * near_high).sum(axis=0) / near_high.sum(axis=0)
    swing_lo = ( party_seat_shares * near_low ).sum(axis=0) / near_low.sum(axis=0)
    swing_emp = (swing_hi - swing_lo) / window
    return swing_emp

def swing_slope(party_seat_shares, party_vote_shares):
    """
    Computes the swing ratio as the slope of the regression of seat shares won by a party onto vote shares won by a party.

    Arguments
    ---------
    party_seat_shares   :   np.ndarray (k,p)
                            array containing the fraction of overall seats party j wins, j = 1, 2, ... , p over k replications
    party_vote_shares   :   np.ndarray (k,p)
                            array containing the faction of overall votes party j wins, j = 1, 2, ..., p over k replications

    Returns
    -------
    lm_slopes           :   np.ndarray (p,)
                            the slopes of the j'th seat share predicted by the j'th vote share.
    lm_resids           :   np.ndarray (p,)
                            the sum of squared residuals in the regression estimated, seat_share ~ vote_share + const
    """
    k,p = party_vote_shares.shape
    ### Linear model coefficient of simulated seat share on simulated vote shares
    lms = [np.polyfit(party_vote_shares.T[:,j], party_seat_shares.T[:,j], 1, full=True)
            for j in range(p)]

    coeffs, lm_resids, _a,_b,_c = zip(*lms)
    lm_slopes = np.asarray([coef[0] for coef in coeffs])
    lm_resids = np.asarray(lm_resids).flatten()
    return lm_slopes, lm_resids

def intervals(party_seat_shares, party_vote_shares, percentiles=[5, 95]):
    """
    computes the confidence interval in the space of observed/simulated vote shares containined in party_vote_shares, party_seat_shares.

    Arguments
    ----------
    party_seat_shares   :   np.ndarray (n,p)
                            array containing the fraction of overall seats party j wins, j = 1, 2, ... , p over n replications
    party_vote_shares   :   np.ndarray (n,p)
                            array containing the faction of overall votes party j wins, j = 1, 2, ..., p over n replications
    percentiles         :   array-like (2,)
                            percentiles to use for the confidence intervals. should be provided in terms of 100*X%, rather than as the decimals themselves.

    Returns
    --------
    all_intervals, (k,3) where the first column contains each of the percentage points between the minimum and maximum party_vote_share values. The second column contains the lower percentile and the third column contains the upper precentile.
    """
    sim_point_range = (np.floor(party_vote_shares.min(axis=0)*100),
                       np.ceil(party_vote_shares.max(axis=0)*100) )
    vote_pcts_covered = [np.arange(*range_) for range_ in zip(*sim_point_range)]
    all_intervals = []
    for j, range_ in enumerate(vote_pcts_covered):
        intervals = []
        for k, vote_pct_j in enumerate(range_):
            filt = np.logical_and(party_vote_shares[:,j] >= (vote_pct_j/100.0 - .002),
                                  party_vote_shares[:,j] < (vote_pct_j/100.0+.002))
            this_shares = party_seat_shares[filt, j]
            if this_shares.size < 10: #inherit linzer's default
                intervals.append(np.hstack((vote_pct_j, np.ones(3,)*np.nan)))
            else:
                mean = this_shares.mean()
                ptiles = np.percentile(this_shares, percentiles)
                intervals.append(np.hstack((vote_pct_j, mean, *ptiles)))
        all_intervals.append(np.asarray(intervals))
    return all_intervals

def winners_bonus(obs_turnout, obs_shares, swing_ratios, rule=ut.plurality_wins):
    """
    Compute the partisan bias between parties by extraploating from their observed vote to the median and computing the excess seat share at median vote.

    That is, given the slope of the seats votes curve, swing_ratio, solve:

    y = swing_ratio ( observed_vote_share - .5) + observed_seat_share

    which provides the expected seat share at voteshare = .5

    Then, this share - .5 yields the expected "fair" share between the parties.

    Arguments
    ----------
    turnout         : np.ndarray (n,1)
                      vector containing turnouts in each district
    obs_shares      : np.ndarray (n,p)
                      observed share of vote won by party j, j = 1, 2, ..., p in each district
    swing_ratios    : np.ndarray (p,)
                      estmate of the slope seats votes curve for parties j \in p
    rule            : callable(a:np.array(N,p)) -> np.array(N,1)
                      callable that reduces the (N,p) array of vote counts into a vector correctly classifying which party won in each seat. The way the factor is sorted in the return vector must match the order of the columns returned by np.unique
    Returns
    --------
    upper triangle of bias matrix, which corresponds to the extra seats won by the jth party against the kth party, k != j, when those two parties split the vote evenly at 50\%.

    In addition, this returns the expected seat share won when the party recieves 50% vote share.

    NOTE: this is somewhat suspect for multiparty systems, since it's not always feasible for two parties to split at 50%, and indeed, it's often unlikely that this occurs. Linzer (2012) suggests that this only makes sense for multiparty models when the coverage of the simulations includes the party at 50%.
    """
    obs_votes = obs_turnout * obs_shares
    obs_grand_voteshare = obs_votes.sum(axis=0) / obs_votes.sum()
    obs_winners = rule(obs_votes).reshape(-1,1)
    _, obs_seats = np.unique(obs_winners, return_counts=True)
    obs_grand_seatshare = obs_seats / obs_seats.sum()
    extrap = (np.ones_like(swing_ratios)*.5 - obs_grand_voteshare) * swing_ratios
    extrap += obs_grand_seatshare
    biases = extrap - .5
    return biases, extrap

def pairwise_winners_bonus(obs_turnout, obs_shares, swing_ratios, rule=ut.plurality_wins):
    """
    Compute the pairwise partisan bias, or the difference between the seat share that a party would win if it split the vote 50/50 with another party.

    The extrapolation is documented in winners_bonus. The difference between pairwise and direct is that this examines the difference in projected seat shares each party would get at 50\% vote share.

    Arguments
    ----------
    turnout         : np.ndarray (n,1)
                      vector containing turnouts in each district
    obs_shares      : np.ndarray (n,p)
                      observed share of vote won by party j, j = 1, 2, ..., p in each district
    swing_ratios    : np.ndarray (p,)
                      estmate of the slope seats votes curve for parties j \in p
    rule            : callable(a:np.array(N,p)) -> np.array(N,1)
                      callable that reduces the (N,p) array of vote counts into a vector correctly classifying which party won in each seat. The way the factor is sorted in the return vector must match the order of the columns returned by np.unique
    Returns
    --------
    upper triangle of bias matrix, which corresponds to the extra seats won by the jth party against the kth party, k != j, when those two parties split the vote evenly at 50\%.

    In addition, this returns the expected seat share won when the party recieves 50% vote share.

    NOTE: this is somewhat suspect for multiparty systems, since it's not always feasible for two parties to split at 50%, and indeed, it's often unlikely that this occurs. Linzer (2012) suggests that this only makes sense for multiparty models when the coverage of the simulations includes the party at 50%.
    """
    _, extrap = winners_bonus(obs_turnout, obs_shares, swing_ratios, rule=rule)
    return np.triu(np.subtract.outer(extrap, extrap)), extrap

def attainment_gap(obs_turnout, obs_shares, swing_ratios, rule=ut.plurality_wins):
    """
    Compute the attainment bias, or the difference between the vote shares required to achieve an outright majority in the legislature and 50%.

    This is an alternative measure of partisan bias to the winners bonus, which measures the "extra" seats won by parties when parties win exactly 50% of the vote. In theory, this provides the relative difference in how difficult it is for a party to control the legislature, whereas the winners bonus measure indicates the party who would be favored to control the legislature if both parties were equally preferred.

    Arguments
    ----------
    turnout         : np.ndarray (n,1)
                      vector containing turnouts in each district
    obs_shares      : np.ndarray (n,p)
                      observed share of vote won by party j, j = 1, 2, ..., p in each district
    swing_ratios    : np.ndarray (p,)
                      estmate of the slope seats votes curve for parties j \in p
    rule            : callable(a:np.array(N,p)) -> np.array(N,1)
                      callable that reduces the (N,p) array of vote counts into a vector correctly classifying which party won in each seat. The way the factor is sorted in the return vector must match the order of the columns returned by np.unique
    Returns
    --------
    the attainment biases, which corresponds to the difference between .5 and the attainment threshold, the vote share projected to gain 50% of the legislature. Is negative if a party must win more than 50% of the vote to get 50% of the legislature.

    In addition, this returns the expected vote share for parties at 50\% seat share.

    NOTE: this is somewhat suspect for multiparty systems, since it's not always feasible for two parties to split at 50%, and indeed, it's often unlikely that this occurs. Linzer (2012) suggests that this only makes sense for multiparty models when the coverage of the simulations includes the party at 50%.
    """
    obs_turnout = obs_turnout.reshape(-1,1)
    obs_shares = obs_shares.reshape(-1,2)
    obs_votes = obs_turnout * obs_shares
    obs_grand_voteshare = obs_votes.sum(axis=0) / obs_votes.sum()
    obs_winners = rule(obs_votes).reshape(-1,1)
    N,p = obs_shares.shape
    _, obs_seats = np.unique(obs_winners, return_counts=True)
    obs_grand_seatshare = obs_seats / obs_seats.sum()
    to_median_seats = (np.ones_like(swing_ratios) * .5 - obs_grand_seatshare) / swing_ratios
    to_median_seats += obs_grand_voteshare
    biases = to_median_seats - .5
    return biases, to_median_seats

def pairwise_attainment_gap(obs_turnout, obs_shares, swing_ratios, rule=ut.plurality_wins):
    """
    Compute the upper triangle of the pairwise attainment bias matrix, or the difference between the vote shares required to achieve an outright majority or all parties. This provides the relative difference in attainment thresholds, and is negative when the party corresponding to the row of the matrix has a lower attainment threshold than the party in the column of the matrix.

    Arguments
    ----------
    turnout         : np.ndarray (n,1)
                      vector containing turnouts in each district
    obs_shares      : np.ndarray (n,p)
                      observed share of vote won by party j, j = 1, 2, ..., p in each district
    swing_ratios    : np.ndarray (p,)
                      estmate of the slope seats votes curve for parties j \in p
    rule            : callable(a:np.array(N,p)) -> np.array(N,1)
                      callable that reduces the (N,p) array of vote counts into a vector correctly classifying which party won in each seat. The way the factor is sorted in the return vector must match the order of the columns returned by np.unique
    Returns
    --------
    the pairwise attainment biases, which corresponds to the difference between each party's attainment thresholds, the vote share required to gain 50% of the legislature. Is negative if a party must win *less* than another party to win control of the legislature. Thus, negative indicates bias in favor of the party, since attainment is *easier* for them than it is for another party.

    In addition, this returns the expected vote share for parties at 50\% seat share.

    This should always be applicable, if the extrapolation to the electoral median makes sense.
    """
    _, extrap = attainment_gap(obs_turnout, obs_shares, swing_ratios, rule=rule)
    biases = np.subtract.outer(extrap, extrap)
    return np.triu(biases), extrap

def directed_waste(voteshares, turnout=None):
    """
    Compute the directed waste in votes over districts in a two-party system. 
    Expresses the percentage of all votes wasted for the reference party over and above the other party. 
    
    If negative, indicates bias against the reference party, in that more votes are wasted by the reference party than by the opponent.

    Arguments
    ----------
    voteshares  :   np.ndarray (n,1)
                    Voteshares for the reference party against which the 
                    measure is computed. 
    turnout     :   np.ndarray (n,1)
                    vector of raw votes cast for all parties in `n` districts
    
    This is the McGhee (2014) measure of waste, where waste is defined by the 
    sum of a party's surplus and losing votes. So, waste for one party is:

    Sum  (votes in districts where party voteshare < .5)
         + ((votes in districts where party voteshare > .5) 
           - (.5 * turnout in districts where party voteshare > .5))
    """
    voteshares = np.asarray(voteshares).reshape(-1,1)
    if turnout is None:
        turnout = np.ones_like(voteshares)
    turnout = np.asarray(turnout).reshape(voteshares.shape)
    if not ((0 <= voteshares) & (voteshares <= 1)).all():
        if ((0 <= voteshares) & (voteshares <= 100)).all():
            voteshares /= 100
        else:
            voteshares.dump('failed.array')
            raise Exception('Vote Shares must be between 0 and 1 with no NaN')
    # waste is 
    waste_for = ( ((voteshares > .5) * ((voteshares - .5) * turnout)) #in excess of victory
                 +((voteshares < .5) * (voteshares * turnout))).sum() #cast for losing candidates
    waste_against = ( ((voteshares < .5) * ((voteshares - .5) * turnout))
                     +((voteshares > .5) * (voteshares * turnout))).sum()
    return waste_against - waste_for

def efficiency_gap(voteshares, turnout=None):
    """
    compute the efficiency gap, which is defined as the directed waste divided by the total votes cast in an election
    """
    if turnout is None:
        sbar = (voteshares > .5).mean()
        hbar = voteshares.mean()
        return (sbar - .5) - 2 * (hbar - .5)
    else:
        dw = directed_waste(voteshares, turnout)
        return dw/np.sum(turnout)
