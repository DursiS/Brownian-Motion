import Distributions
import matplotlib.pyplot as plt
import math
import numpy as np


class BrownianMotion:
    """A position evolving by Gaussian Increments

    Public Attributes:
        - position: The net change of all steps from origin 0.0
        - h_step: How much height changes for each time_step taken
        - time_step: How much time it takes for each step
        - path: A history of all positions, right-most being current position

    Private Attributes:
        - _steps_taken: The total number of steps taken
    """

    position: tuple[float, float]
    h_step: float
    time_step: float
    path: list[tuple[float, float]]
    steps_taken = int

    def __init__(
        self,
        h_step: float,
        time_step: float,
        mu: float = 0,
        sigma: float = 1,
    ) -> None:
        """Create a new BrownianMotion.

        Preconditions:
            - h_step > 0
            - time_step > 0
        """
        self.position = 0.0, 0.0
        self.h_step = h_step
        self.time_step = time_step
        self.path = [(0, 0)]
        self.mu = mu
        self.sigma = sigma
        self.steps_taken = 0

    def step(self) -> None:
        """Move one time_step forward"""

        random_shock = np.random.normal(self.mu, self.sigma)
        drift = self.mu * self.time_step
        noise = self.sigma * (self.time_step ** (1 / 2)) * random_shock
        step = drift + noise

        new_pos = (self.position[0] + self.time_step), (self.position[1] + step)
        self.position = new_pos
        self.path.append(new_pos)
        self.steps_taken += 1

    def run(self, n: int) -> None:
        """Move <n> unit-times forward"""
        for i in range(n):
            self.step()

    def expectation(self) -> float:
        """Return the expected final height given
        len(self.steps) steps were taken."""

        time_passed = self.steps_taken * self.time_step
        return self.mu * time_passed

    def expt_confidence_interval(self) -> tuple[float, float]:
        """Return an interval in which the True/Theoretical
        expected value is almost surely to be.
        (i.e. within 3 Std Dev.)"""

        expt, std = self.expectation(), self.std()
        return expt - (3 * std), expt + (3 * std)

    def var(self) -> float:
        """Return the variance in the endpoint of this BrownianMotion."""

        x1 = [point[1] for point in self.path]
        mean = sum(x1) / len(x1)
        x2 = [point**2 for point in x1]
        mean_sq = sum(x2) / len(x2)
        return mean_sq - mean

    def std(self) -> float:
        """Return the standard deviation of this random walk"""
        return self.var() ** (1 / 2)

    def visualize(self) -> None:
        """Plot this BrownianMotion."""
        x = [item[0] for item in self.path]
        y = [item[1] for item in self.path]
        name = self.__class__.__name__
        plt.plot(x, y, lw=0.5, alpha=0.5, label=name)

    def visualize_mean(self) -> None:
        """Plot the stats for the <number>th RandomWalk after <n> steps."""
        # Mean
        x1 = [item[0] for item in self.path]
        expt = self.expectation()
        y2 = [expt for i in range(self.steps_taken)]
        name = self.__class__.__name__
        plt.plot(x1[:-1], y2, c="#00008B", label=f"E(X) {name}")

    def visualize_root(self) -> None:
        """Plot -+Root(n) as 2 constant linear equations
        to have an idea where this BrownianMotion will lie.
        """
        x1 = [item[0] for item in self.path]
        time_passed = self.steps_taken * self.time_step
        y2 = [time_passed ** (1 / 2) for i in range(self.steps_taken)]
        y3 = [-(time_passed ** (1 / 2)) for i in range(self.steps_taken)]
        plt.plot(x1[:-1], y2, c="#FF0000", label="Root(t)")
        plt.plot(x1[:-1], y3, c="#FF0000")

    def visualize_variance_std(self) -> None:
        """Add var and std to the legend."""
        raise NotImplementedError

    def value_at_risk(self) -> float:
        """Return the Value-At-Risk (VaR) of this BrownianMotion"""
        return np.percentile(self.path, 95)


def copy_bm(bm: BrownianMotion) -> BrownianMotion:
    """Return a copy of this BrownianMotion."""
    new_bm = BrownianMotion(bm.h_step, bm.time_step)
    new_bm.path = bm.path[:]
    new_bm.steps_taken = bm.steps_taken
    new_bm.mu = bm.mu
    new_bm.sigma = bm.sigma
    return new_bm


# def estimate_probability(bm: BrownianMotion) -> float:
#     """Return the realized probability of stepping up."""
#     total = 0
#     for point in bm.path:
#         if point[1] > 0:
#             total += 1
#     return total / len(bm.path)
#
#
# def p_confidence_interval(m: int, n: int, dt: float, dz: float) -> tuple[float, float]:
#     """Return an interval in which the True/Theoretical
#     probability of stepping up is almost surely to be.
#     By doing <k> Monte Carlo Simulations to approximate p.
#     Each Simulation
#     Which by LLN itself and it's variance convergences for large n."""
#
#     matrix = get_path_matrix(m, n, dt, dz)
#     total = 0
#     for bm in matrix:
#         total += estimate_probability(bm)
#     average = total / len(matrix)


def get_average_point(
    i: int, matrix: list[list[tuple[float, float]]]
) -> tuple[float, float]:
    """Helper function to get_average_path
    Return the average of point <i> of paths in <matrix>,
    starting from 0 for the origin.
    """
    total = 0
    for path in matrix:
        total += path[i][1]
    return matrix[0][i][0], (total / len(matrix[0][0]))


def get_average_path(
    matrix: list[list[tuple[float, float]]]
) -> list[tuple[float, float]]:
    """Return the average path of Brownian Motions of this matrix,"""
    average = []
    for i in range(len(matrix[0])):
        average.append(get_average_point(i, matrix))
    return average[1:]


def get_path_matrix(
    m: int, n: int, dt: float, dz: float
) -> list[list[tuple[float, float]]]:
    """Return a (m x n) matrix with m paths, each with n steps."""

    matrix = []
    for i in range(m):
        bmi = BrownianMotion(dt, dz)
        bmi.run(n)
        matrix.append(bmi.path)
    return matrix


def get_average_pnl(matrix: list[list[tuple[float, float]]]) -> float:
    """Get the average PNL of endpoints in <matrix>.

    Precondition: len(matrix) > 0
    """

    total = 0
    for path in matrix:
        total += path[-1][1]
    return round(total / len(matrix), 4)


def get_expected_shortfall(matrix: list[list[tuple[float, float]]]) -> float:
    """Return the Expected Shortfall of this BrownianMotion,
    as the average of the worst losses along these paths.

    Precondition: There's at least one endpoint below or equal to VaR
    """
    endpoints = [path[-1][1] for path in matrix]
    var = np.percentile(endpoints, 95)
    total, i = 0, 0
    for end in endpoints:
        if end <= var:
            total += end
            i += 1
    return round(total / i, 4)


class GBrownianMotion(BrownianMotion):

    def __init__(
        self,
        h_step: float,
        time_step: float,
        mu: float = 0,
        sigma: float = 1,
    ) -> None:
        """Create a new BrownianMotion.

        Preconditions:
            - h_step > 0
            - time_step > 0
        """
        super().__init__(h_step, time_step, mu, sigma)
        self.position = 0, 1

    def step(self) -> None:
        """Move one time_step forward"""

        shock = np.random.normal(self.mu, self.sigma)
        noise = self.sigma * (self.time_step ** (1 / 2)) * shock
        step_factor = math.exp(
            (self.mu - (1 / 2) * (self.sigma**2)) * self.time_step + noise
        )

        new_pos = (self.position[0] + self.time_step), (self.position[1] * step_factor)
        self.position = new_pos
        self.path.append(new_pos)
        self.steps_taken += 1


if __name__ == "__main__":
    _dt, _dz = 0.001, 0.001
    _n = 1000

    if False:  # Simulate an average path
        paths = get_path_matrix(100, _n, _dt, _dz)
        y = get_average_path(paths)
        x = [i * _dt for i in range(_n)]
        plt.plot(x, y, label="Average")
        plt.show()

    if False:  # Compare a GBM to BM
        bm = BrownianMotion(_dt, _dz)
        bm.run(_n)
        bm.visualize()
        bm.visualize_mean()

        gbm = GBrownianMotion(_dt, _dz)
        gbm.run(_n)
        gbm.visualize()
        gbm.visualize_mean()

        plt.legend()
        plt.show()

    if False:  # Plot 20 BM paths
        for i in range(20):
            bmi = BrownianMotion(_dt, _dz)
            bmi.run(_n)
            bmi.visualize()
        plt.show()

    if False:  # Verify Variance and Mean
        expected_std = (bm.mu * _n) ** (1 / 2)
        print(f"Got Var: {round(bm.var(), 4)}, Expected: {bm.mu * _n})")
        print(f"Got Std Dev: {round(bm.std(), 4)}, Expected: {expected_std}")

    if False:  # Average PNL and Expected Shortfall of <m> paths
        m = 100
        paths = get_path_matrix(100, _n, _dt, _dz)
        pnl = get_average_pnl(paths)
        es = get_expected_shortfall(paths)
        print(f"Got PNL: {pnl}, Expected: {bm.mu * _n}")
        print(f"Expected Shortfall: {es}")
