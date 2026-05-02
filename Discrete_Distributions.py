import math
import random
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm


class RandomVariable:
    """A function mapping Outcomes -> Numbers."""

    def sample(self) -> float:
        """Return one draw from the distribution"""
        raise NotImplementedError

    def pmf(self, k: int):
        """Return P(X = k)"""
        raise NotImplementedError

    def maximize(self, k: int) -> float:
        """Return the variable which will maximize pmf for <k>."""
        raise NotImplementedError


class Binomial(RandomVariable):
    """A RV following a Binomial Distribution

    Values and their probabilities are distributed like the amount of
    successes within n number of trials.
    """

    def __init__(self, n: int, theta: float) -> None:
        self.n = n
        self.theta = theta

    def sample(self) -> int:
        """Return one draw from the distribution"""
        count = 0
        for i in range(self.n):
            if random.random() < self.theta:
                count += 1
        return count

    def pmf(self, k: int) -> float:
        """Return the probability of X = k."""
        combinations_of_k = math.comb(self.n, k)
        losing_the_rest = (1 - self.theta) ** (self.n - k)
        winning_k = self.theta**k
        return combinations_of_k * losing_the_rest * winning_k

    def maximize(self, k: int) -> float:
        """Differentiate pmf(k) and solve for what theta maximizes it."""
        raise NotImplementedError


class Geometric(RandomVariable):
    """A RV following a Geometric Distribution

    Values and their probabilities are distributed
    like the wait time of a success.
    """

    def __init__(self, theta: float) -> None:
        self.theta = theta

    def sample(self) -> int:
        """Return one draw from the distribution"""
        count = 1
        while random.random() >= self.theta:
            count += 1
        return count

    def pmf(self, k: int) -> float:
        """Return the probability of this RV being exactly k."""
        return ((1 - self.theta) ** k) * self.theta

    def pmf_range(self, k, start=None, stop=None) -> float:
        """Return the probability of this RV being <above> and/or
        below <stop> inclusive.
        """

        fail_before = (1 - self.theta) ** (start - 1)
        fail_after = (1 - self.theta) ** (stop + 1)

        if start is not None and stop is not None:
            return fail_before - fail_after
        elif start is None:
            return fail_before * self.theta
        else:  # end is None
            return 1 - fail_after


class Poisson(RandomVariable):
    """A RV following a Poisson Distribution"""

    def __init__(self, lam: float) -> None:
        self.lam = lam

    def sample(self) -> int:
        """Return how many events occurred in this time period,
        driven by exponential intervals.
        """
        L = math.exp(-self.lam)
        multiplications = 0
        p = 1.0

        while p > L:
            multiplications += 1
            p *= random.random()
        return multiplications

    def pmf(self, k: int) -> float:
        """Return the probability of X = k given
        <self.lam> is the expected valued of X."""
        return ((self.lam**k) * (math.exp((-1) * self.lam))) / math.factorial(k)

    def maximize(self, k) -> float:
        """The expected value lam which maximizes the probability X = k
        is when lam is that value."""
        return k


class Normal(RandomVariable):
    """A Continuous RV following a Normal Distribution

    Public Attributes:
        - mew: The average
        - gamma: The variation
    """

    mew: float
    gamma: float

    def __init__(self, mu: float = 0, sigma: float = 1):
        self.mu = mu
        self.sigma = sigma

    def sample(self) -> float:
        """Return a random draw"""
        return np.random.normal(self.mu, self.sigma)

    def pmf(self, z: float) -> float:
        """Return the probability of X = z"""
        return norm.pdf(z, self.mu, self.sigma)


if __name__ == "__main__":
    # Choose your experiment:
    # 'G': Geometric, 'B': Binomial, 'P': Poisson
    experiment = "P"
    theta = 1 / 3
    trials = 1000
    n = 10
    lam = 3

    ########################################################################
    # Geometric Distribution Plotting
    ########################################################################
    if experiment == "G":

        geo = Geometric(theta)
        geo_samples = [geo.sample() for i in range(trials)]
        geo_probabilities = [geo.pmf(sample) for sample in geo_samples]

        plt.hist(
            geo_samples,
            bins=np.arange(0, max(geo_samples) + 2) - 0.5,
            density=True,
            alpha=0.5,
        )
        plt.stem(geo_samples, geo_probabilities)
        plt.title(f"Empirical Geometric Distribution given {trials} trials")
        plt.xlabel("Count until success")
        plt.ylabel("Probability 0.00-1.00")
        plt.show()

    ########################################################################
    # Binomial Distribution Plotting
    ########################################################################
    elif experiment == "B":

        bio = Binomial(n, theta)
        bio_samples = [bio.sample() for j in range(trials)]
        bio_probabilities = [bio.pmf(sample) for sample in bio_samples]

        plt.hist(
            bio_samples,
            bins=np.arange(0, max(bio_samples) + 2) - 0.5,
            density=True,
            alpha=0.5,
        )
        plt.stem(bio_samples, bio_probabilities)
        plt.title(f"Empirical Binomial Distribution given {trials} trials")
        plt.xlabel("Count until success")
        plt.ylabel("Probability 0.00-1.00")
        plt.show()

    ########################################################################
    # Poisson Distribution Plotting
    ########################################################################
    elif experiment == "P":

        poi = Poisson(lam)
        poi_samples = [poi.sample() for i in range(trials)]
        poi_probabilities = [poi.pmf(sample) for sample in poi_samples]

        plt.hist(
            poi_samples,
            bins=np.arange(0, max(poi_samples) + 2) - 0.5,
            density=True,
            alpha=0.5,
        )
        plt.stem(poi_samples, poi_probabilities)
        plt.title(f"Empirical Poisson Distribution given {trials} trials")
        plt.xlabel("Count until success")
        plt.ylabel("Probability 0.00-1.00")
        plt.show()
