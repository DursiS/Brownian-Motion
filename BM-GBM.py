import Discrete_Distributions
import matplotlib.pyplot as plt
import math
import numpy as np


class BrownianMotion:
    """A position evolving by Gaussian Increments

    Public Attributes:
        - position: The net change of all steps from origin 0.0
        - theta: The probability of stepping up
        - h_step: How much height changes for each time_step taken
        - time_step: How much time it takes for each step
        - path: A history of all positions, right-most being current position

    Private Attributes:
        - _steps_taken: The total number of steps taken
    """

    position: tuple[float, float]
    theta: float
    h_step: float
    time_step: float
    path: list[tuple[float, float]]
    _steps_taken = int

    def __init__(
        self,
        theta: float,
        h_step: float,
        time_step: float,
        mu: float = 0,
        sigma: float = 1,
    ) -> None:
        """Create a new BrownianMotion.

        Preconditions:
            - h_step > 0
            - time_step > 0
            - theta > 0
        """
        self.position = 0.0, 0.0
        self.theta = theta
        self.h_step = h_step
        self.time_step = time_step
        self.path = [(0, 0)]
        self.mu = mu
        self.sigma = sigma
        self._steps_taken = 0

    def step(self) -> None:
        """Move one time_step forward"""

        random_shock = np.random.normal(self.mu, self.sigma)
        drift = self.mu * self.time_step
        noise = self.sigma * (self.time_step ** (1 / 2)) * random_shock
        step = drift + noise

        new_pos = (self.position[0] + self.time_step), (self.position[1] + step)
        self.position = new_pos
        self.path.append(new_pos)
        self._steps_taken += 1

    def run(self, n: int) -> None:
        """Move <n> unit-times forward"""
        for i in range(n):
            self.step()

    def expectation(self) -> float:
        """Return the expected final height given
        len(self.steps) steps were taken."""

        time_passed = self._steps_taken * self.time_step
        return self.mu * time_passed

    def var(self) -> float:
        """Return the variance of this random walk"""
        pass

    def std(self) -> float:
        """Return the standard deviation of this random walk"""
        pass

    def visualize(self) -> None:
        """Plot this BrownianMotion."""
        x = [item[0] for item in self.path]
        y = [item[1] for item in self.path]
        plt.plot(x, y)

    def visualize_stats(self) -> None:
        """Plot the stats for the <number>th RandomWalk after <n> steps."""
        # Mean
        x1 = [item[0] for item in self.path]
        expt = self.expectation()
        y2 = [expt for i in range(self._steps_taken)]
        name = self.__class__.__name__
        plt.plot(x1[:-1], y2, c="#00008B", label=f"E(X) {name}")

        # Root and Legend
        time_passed = self._steps_taken * self.time_step
        y2 = [time_passed ** (1 / 2) for i in range(self._steps_taken)]
        y3 = [-(time_passed ** (1 / 2)) for i in range(self._steps_taken)]
        plt.plot(x1[:-1], y2, c="#FF0000", label="Root(t)")
        plt.plot(x1[:-1], y3, c="#FF0000")


class GBrownianMotion(BrownianMotion):

    def __init__(
        self,
        theta: float,
        h_step: float,
        time_step: float,
        mu: float = 0,
        sigma: float = 1,
    ) -> None:
        """Create a new BrownianMotion.

        Preconditions:
            - h_step > 0
            - time_step > 0
            - theta > 0
        """
        super().__init__(theta, h_step, time_step, mu, sigma)
        self.position = 0, 1

    def step(self) -> None:
        """Move one time_step forward"""

        shock = Discrete_Distributions.Normal(0, 1)
        noise = self.sigma * (self.time_step ** (1 / 2)) * shock.sample()
        step_factor = math.exp(
            (self.mu - (1 / 2) * (self.sigma**2)) * self.time_step + noise
        )

        new_pos = (self.position[0] + self.time_step), (self.position[1] * step_factor)
        self.position = new_pos
        self.path.append(new_pos)
        self._steps_taken += 1


if __name__ == "__main__":
    dt, dz = 0.001, 0.001
    theta = 1 / 2
    n = 10000

    bm = BrownianMotion(theta, dt, dz)
    bm.run(n)
    bm.visualize()
    bm.visualize_stats()

    gbm = GBrownianMotion(theta, dt, dz)
    gbm.run(n)
    gbm.visualize()
    gbm.visualize_stats()

    plt.legend()
    plt.show()
