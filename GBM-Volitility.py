import matplotlib.pyplot as plt
import scipy
from numpy import ndarray, dtype, float64

import math
from BM import BrownianMotion
import numpy as np
import yfinance as yf


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


def compute_risk(data: ndarray) -> tuple[float, float]:
    """Return the VaR and ES of paths in <data>"""
    pnl = data[-1]
    losses = -pnl
    var_95 = np.percentile(losses, 95)
    es = losses.mean()

    return var_95, es


def get_returns(ticker: str) -> ndarray:
    """Return the annual returns of <ticker>

    Precondition: <ticker> is a valid Stock Ticker
    """
    data = yf.download(ticker, period="1y")
    prices = data["Close"].iloc[:, 0]
    return np.log(prices / prices.shift(1))


def get_realized_vol(returns: ndarray) -> float:
    """Return the annual standard deviation of <ticker>"""
    rv = np.std(returns) * np.sqrt(252)
    return round(rv, 4)


def get_price_call(returns: ndarray, K: float, r: float, T: float) -> float:
    """Return the expected value of a call of strike <K>,
    given risk-free interest rate <r> and time <T>.
    """
    payoffs = np.maximum(returns - K, 0)
    return np.mean(payoffs) * np.exp(-r * T)


def black_scholes_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    S: current stock price
    K: strike price
    T: time to expiry in years
    r: risk-free interest rate
    sigma: volatility
    """
    d1 = (math.log(max(S / K, 0.0000001)) + (r + 0.5 * sigma**2) * T) / (
        sigma * math.sqrt(T)
    )
    return d1


def get_implied_vol(returns: ndarray, K: float, r: float, T: float) -> float:
    """Return the implied volatility sigma,
    such that ModelPrice(sigma) = ObservedPrice."""

    realized_vol = get_realized_vol(returns)
    d1 = np.array([black_scholes_d1(S, K, T, r, realized_vol) for S in returns[1:]])
    d1 = np.array(list(d1))
    vol = np.linspace((realized_vol * 0.8), (realized_vol * 1.2), 250)

    model_prices = K * np.exp(d1 * vol * np.sqrt(T) - (r + 0.5 * vol**2) * T)
    dp = np.array(returns[1:]) - model_prices

    iv = min(vol[np.abs(dp) < 0.1])
    return round(iv, 4)


if __name__ == "__main__":
    tick = "SPY"
    _dt = 1 / 252

    if True:  # Compare RV to IV
        returns = get_returns(tick)
        rv = get_realized_vol(returns)
        iv = get_implied_vol(returns, 500, 0.1, 1)
        print(f"IV: {iv}, RV: {rv}")

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
