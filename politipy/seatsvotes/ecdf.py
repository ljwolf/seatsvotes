import numpy as np

class Empirical_CDF(object):
    """
    This class mimics the seats votes curve estimation done in the R package, `pscl`.
    """
    def __init__(self, votes, k=1001):
        """
        Arguments
        ----------
        votes   :   array-like
                    vote shares won by a reference party
        k       :   int
                    number of divisions to use for the ecdf estimate
        method  :   str
                    fixed to 'ecdf' at the moment.
        """
        votes = np.array(votes)
        self.support = np.linspace(-1, 1, num=k)
        self._n_elex = len(votes)
        self._original_votes = votes
        self.votes = votes
        self._n_contested = len(self.votes)
        self.seats = (self.votes > .5).astype(int)

        self.est_votes, self.est_seats = self._compute()
        self.votes = 1 - self.votes
        self._opp_votes, self._opp_seats = self._compute()
        self.votes = self._original_votes

    def _compute(self):
        est_votes = []
        est_seats = []
        for i in range(len(self.support)):
            vote_diff = self.votes - self.support[i]
            est_votes.append(np.mean(vote_diff))
            seat_share = ((self.votes - self.support[i]) > .5)
            est_seats.append(np.mean(seat_share))
        est_votes = np.asarray(est_votes)
        est_seats = np.asarray(est_seats)
        inbounds = (est_votes >= 0) & (est_votes <= 1)
        return est_votes[inbounds], est_seats[inbounds]

    def asymmetry_at(self, share):
        hinge = np.where(self.ecdf.est_votes >= share)[0].max()
        ref_share = ecdf.est_seats[hinge]
        opp_share = ecdf._opp_seats[hinge]
        return ref_share - opp_share
