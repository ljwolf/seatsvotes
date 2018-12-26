from collections import Counter, OrderedDict, namedtuple
import pandas as pd
import numpy as np

Pattern = namedtuple("Pattern", ['pattern', 'mask', 'contests', 'count'])

def extract_patterns(df):
    """
    Extracts the unique patterns of contestation in an electoral system. Mimicks Linzer 2012's `findpatterns` R function.

    Arguments
    ----------
    df      :   data frame
                dataframe containing vote values
    votecols:   list
                list of columns containing the subset of the dataframe to use for computing the patterns
    Returns
    --------
    unique pattern of contestation, which is just the set of unique party vs. party contests.

    Example
    ---------
    If a system has three parties and four districts:
       A   B   C
    0 .5  .3  .2
    1 .2  .8  .0
    2 .2  .5  .3
    3 .9  .1  .0

    then there are two unique patterns of contestation:
    {AB:2, ABC:2}
    """
    votes = df.iloc[:,1:]
    totals = df.iloc[:,0]
    contesteds = votes > 0
    patterns = Counter([tuple(contestation_pattern) for
                        contestation_pattern in contesteds.values])
    pattern_tuples = []
    for pattern, count in patterns.items():
        name = tuple(np.asarray(votes.columns)[np.asarray(pattern)])
        members = [i for i, row in enumerate(contesteds.values)
                   if tuple(row) == pattern]
        contests_in_pattern = pd.concat((totals.iloc[members,].to_frame(), votes.iloc[members,][list(name)]), axis=1)
        pattern_tuples.append(Pattern(pattern = name, count=count,
                                      contests = contests_in_pattern, mask=pattern))
    return pattern_tuples

def filter_uncontested(votes, threshold):
    """
    Filter elections that are effectively uncontested.
    If `threshold` votes is won by any party,
    the election is considered uncontested.
    """
    effectively_uncontested = votes > threshold
    masked_votes = votes[~effectively_uncontested]
    return masked_votes.dropna()

def make_election_frame(votes, shares=None, party_names=None, margin_idx=None):
    """
    Constructs an election frame from at most two arrays.
    If provided,
    """
    if votes.ndim == 1:
        votes = votes.reshape(-1,1)
    if votes.shape[-1] == 1 and shares is not None:
        votes, shares = votes, shares
    elif votes.shape[-1] > 1 and shares is None:
        if margin_idx is None:
            totals = votes.sum(axis=1).reshape(-1,1)
        else:
            totals = votes[:,margin_idx].reshape(-1,1)
            votes = np.delete(votes, margin_idx, axis=1)
        shares = votes / totals
        votes = totals
    data = np.hstack((votes, shares))
    if party_names is None:
        party_names = ['Party_{}'.format(i) for i in range(data.shape[-1] - 1)]
    return pd.DataFrame(data, columns=['Votes'] + list(party_names))

def make_log_contrasts(elex_frame, unc_threshold=.05, votecols=None, holdout=None):
    """
    Compute the log odds ratio of a collection of votes to a target reference holdout vote.

    Arguments
    ---------
    votes           :   DataFrame of (n elections over p parties)
                        pandas dataframe containing votes to use to construct contrasts
    unc_threshold   :   float
                        threshold of contestation for an election
    votecols        :   list
                        list of names to use to subset `votes`, if `votes` contains more than just vote shares.
    holdout         :   str
                        name of the column to use as a holdout column, the denominator of the odds ratio. Takes the first column of DataFrame by default.
    Returns
    -------
    a dataframe containing the log odds ratio of
    (possibly) less than n elections over p-1 parties
    """
    votes = elex_frame
    if votecols is None:
        votecols = votes.columns
    votes = votes[votecols]
    if holdout is None:
        holdout = votecols[0]

    remaining = votes.drop(holdout, axis=1).values
    holdout_data = votes[[holdout]].values
    out = np.log(remaining / holdout_data)
    out = pd.DataFrame(out, columns=[col for col in votes if col != holdout],
                            index=votes.index)
    out = pd.concat((elex_frame.iloc[:,0], out), axis=1)
    return out


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('./canada1979.csv')
    votecols = ['Liberal', 'Prog. Conservative', 'New Democratic Party', 'Social Credit']
    patterns = extract_patterns(df, votecols=votecols)
    logratios = [make_log_contrasts(pattern.contests,
                                    votecols=votecols,
                                    holdout='Liberal') for pattern in patterns]
