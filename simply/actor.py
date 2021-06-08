import numpy as np
import pandas as pd


class Actor:
    # TODO rename bids/asks to order and use named tuple
    def __init__(self, actor_id, df, ls=1, ps=1):
        # TODO add battery component
        # TODO add individual scale values and scale time series accordingly
        self.id = actor_id
        self.t = 0
        self.load_scale = ls
        self.pv_scale = ps
        self.battery = None
        self.load = ls * df["load"].to_list()
        self.pv = ps * df["pv"].to_list()
        self.prices = df["prices"].to_list()
        # perfect foresight
        self.schedule = df["pv"] - df["load"]
        self.bids = []
        self.traded = {}

    def generate_order(self):
        # TODO calculate amount of energy to fulfil personal schedule
        energy = self.schedule[self.t]
        # TODO simulate strategy: manipulation, etc.
        price = self.prices[self.t]
        # TODO take flexibility into account to generate the bid

        new_bid = (self.t, self.id, abs(energy), price)
        # TODO place bid on market
        self.bids.append(new_bid)
        self.t += 1

        return new_bid, np.sign(energy)


    def receive_market_results(self, time, energy, price):
        # TODO update schedule, if possible e.g. battery
        # TODO post settlement of differences
        assert time < self.t
        post = (energy, price)
        self.traded[time] = post

    def to_dict(self):
        return self.id


def create_random(actor_id):
    # TODO improve random signals
    # TODO add random generation signal (e.g. with PV characteristic)
    nb_ts = 24
    time_idx = pd.date_range("2021-01-01", freq="H", periods=nb_ts)
    cols = ["load", "pv", "prices"]
    values = np.random.rand(nb_ts, len(cols))
    df = pd.DataFrame(values, columns=cols, index=time_idx)

    return Actor(actor_id, df)
