from typing import Any

import matplotlib.pyplot as plt
import scipy
from numpy import ndarray, dtype, float64
from scipy.stats import norm

import math
from BM import BrownianMotion
import numpy as np
import yfinance as yf


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


def get_call_price(prices: ndarray, K: float, r: float, T: float) -> float:
    """Return the expected value of a call of strike <K>,
    given risk-free interest rate <r> and time <T>.
    """
    payoffs = np.maximum(prices - K, 0)
    return np.mean(payoffs) * np.exp(-r * T)


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    S: current stock price
    K: strike price
    T: time to expiry in years
    r: risk-free interest rate
    sigma: volatility
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * (T ** (1 / 2))
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def get_implied_vol(
    C_market: float, S: float, realized_vol: float, K: float, r: float, T: float
) -> float:
    """Return the volatility sigma, such that ModelPrice(sigma) ~ ObservedPrice
    for market price <P> of the call, Strikes price K and given RV <realized_vol>.
    T is in years."""
    vols = np.linspace(0.001, 1, 10000)
    model_call_prices = np.array([black_scholes_call(S, K, T, r, vol) for vol in vols])

    diff = np.abs(C_market - model_call_prices)
    iv = float(vols[np.argmin(diff)])
    return round(iv, 4)


def compute_risk(returns: ndarray) -> tuple[float, float]:
    """Return the VaR and ES of paths in <data>"""
    pnl = returns[-1]
    losses = -pnl
    var_95 = np.percentile(losses, 95)
    es = losses.mean()

    return round(var_95, 4), round(es, 4)


def build_smile(
    strikes: ndarray[float], S: float, realized_vol: float, K: float, r: float, T: float
) -> ndarray:
    """Return a ndarray of implied volatiles for each strike in <strikes."""
    return np.array([get_implied_vol(C, S, realized_vol, K, r, T) for C in strikes])


def plot_iv_smile(ticker: str) -> None:
    """Plot a IV against strike prices for stock <ticker>

    Precondition: <ticker> is a valid stock ticker
    """
    returns = get_returns(ticker)
    rv = get_realized_vol(returns)

    tick = yf.Ticker(ticker)
    expiry = tick.options[0]
    calls = tick.option_chain(expiry).calls
    calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)]
    mid = (calls["bid"] + calls["ask"]) / 2
    mid = mid > 0.5
    strikes = calls["strike"]
    mkt = tick.fast_info["lastPrice"]

    y = build_smile(mid, mkt, rv, mkt, 0.1, 1)
    plt.plot(strikes, y)
    plt.xlabel("Strike Prices")
    plt.ylabel("Implied Volatility")
    plt.show()


def plot_vol_cluster(ticker: str, period: int = 20) -> None:
    """Plot Rolling Volatility of <period> of time against
    returns means of <ticker>

    Precondition: <ticker> is a valid stock ticker.
    """
    returns = get_returns(ticker)
    rolling_stds = np.array(
        [np.std(returns[i : i + period]) for i in range(len(returns) - period + 1)]
    )
    return_means = np.array(
        [np.mean(returns[i : i + period]) for i in range(len(returns) - period + 1)]
    )
    plt.scatter(return_means, rolling_stds)
    plt.xlabel(f"Mean Returns for that {period} day period")
    plt.ylabel("Realized Volatility")
    plt.show()


if __name__ == "__main__":
    tick = "SPY"

    if False:  # Compare RV to computed IV
        rv = 0.125
        iv = get_implied_vol(5)
        print(f"IV: {iv}, RV: {rv}")

    if False:  # Print ES and Var of <tick>
        returns = get_returns(tick)
        var_95, es = compute_risk(returns)
        print(f"VaR: {var_95}, ES: {es}")

    if False:
        plot_iv_smile("SPY")

    if True:
        plot_vol_cluster("SPY")
