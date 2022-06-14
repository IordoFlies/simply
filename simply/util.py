import numpy as np
import pandas as pd


def gaussian_pv(ts_hour, std):
    # One day (24h) time series with peak at noon (12h) and a gaussian curve defined by standard deviation
    x = np.linspace(0, 24 * ts_hour, 24 * ts_hour)
    mean = 12 * ts_hour
    return np.exp(-((x - mean) ** 2) / (2 * std ** 2))


def daily(df, daily_ts=24):
    for i in range(0, len(df), daily_ts):
        yield df.iloc[i:i + daily_ts]


def summerize_actor_trading(sc):
    return (
        pd.DataFrame.from_dict([a.traded for a in sc.actors])
        .unstack()
        .apply(pd.Series)
        .rename({0: "energy", 1: "avg_price"}, axis=1)
    )
