import numpy as np

class SeatsVotes(object):
    """
    This class mimicks the seats votes curve estimation done in the R package, `pscl`.
    """
    def __init__(self, votes, k=1001, method='ecdf'):
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
        self._method = method
        self._n_elex = len(votes)
        self._original_votes = votes
        self.votes = self._validate_votes(votes)
        self._n_contested = len(self.votes)
        self.seats = (self.votes > .5).astype(int)

        self.est_votes, self.est_seats = self._compute()

    def _validate_votes(self, vector):
        filtered = vector[~np.isnan(vector)]
        return filtered

    def _compute(self):
       if self._method is 'ecdf':
           return self._jackman_cdf()

    def _jackman_cdf(self):
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

    @property
    def median_bias(self):
        try:
            return self._median_bias
        except AttributeError:
            hinge = np.argmin(np.abs(self.est_votes - .5))
            bias = self.est_seats[hinge] - .5
            self._median_bias = bias
            return self._median_bias

    def bias_at(self, share):
        hinge = np.argmin(np.abs(self.est_votes - share))
        return self.est_seats[hinge] - share
