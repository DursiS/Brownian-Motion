if __name__ == "__main__":
    tick = "SPY"
    _dt = 1 / 252

    if True:  # Compare RV to IV
        returns = get_returns(tick)
        rv = get_realized_vol(returns)
        iv = get_implied_vol(returns, 500, 0.1, 1)
        print(f"IV: {iv}, RV: {rv}")
