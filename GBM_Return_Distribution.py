import matplotlib.pyplot as plt
from numpy import ndarray, dtype, float64

from BM import BrownianMotion
import numpy as np


class GBrownianMotion(BrownianMotion):

    def __init__(
        self,
        time_step: float,
        mu: float = 0,
        sigma: float = 1,
    ) -> None:
        """Create a new BrownianMotion.

        Preconditions:
            - h_step > 0
            - time_step > 0
        """
        super().__init__(time_step, mu, sigma)

    def simulate_paths(
        self, time: float = 1, n: int = 1
    ) -> ndarray[tuple[int, int], dtype[float64]]:
        """Return an array of <n> simulations of <time> duration."""

        steps = int(time / self.dt)
        paths = np.zeros((steps, n))
        paths[0] = 1

        for i in range(1, steps):
            z = np.random.normal(0, 1, n)
            for j in range(n):
                noise = self.sigma * (self.dt ** (1 / 2)) * z[j]
                step_factor = np.exp(
                    (self.mu - (1 / 2) * (self.sigma**2)) * self.dt + noise
                )

                paths[i, j] = paths[i - 1, j] * step_factor
        return paths


if __name__ == "__main__":
    tick = "SPY"
    _dt = 1 / 252

    if False:  # Simulate return distribution
        returns = get_returns(tick)
        gbm = GBrownianMotion(
            _dt,
            returns.mean(),
            get_realized_vol(returns),
        )

        data = gbm.simulate_paths(1, 10000)
        plt.hist(data[-1], bins=30)
        plt.ylabel("Frequency")
        plt.xlabel("YTD Returns")
        plt.show()
