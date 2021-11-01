import json
from networkx.readwrite import json_graph
import pandas as pd
import random
import glob
import os
import numpy as np

from simply.util import gaussian_pv
from simply import actor
from simply import power_network


class Scenario:
    def __init__(self, network, actors, map_actors, rng_seed=None):
        self.rng_seed = rng_seed if rng_seed is not None else random.getrandbits(32)
        random.seed(self.rng_seed)

        self.power_network = network
        self.actors = list(actors)
        # maps node ids to actors
        self.map_actors = map_actors

    def from_config(path):
        pass

    def __str__(self):
        return "Scenario(network: {}, actors: {}, map_actors: {})".format(
            self.power_network, self.actors, self.map_actors
        )

    def to_dict(self):
        return {
            "rng_seed": self.rng_seed,
            "power_network": {self.power_network.name: self.power_network.to_dict()},
            "actors": {a.id: a.to_dict() for a in self.actors},
            "map_actors": self.map_actors,
        }

    def save(self, dirpath):
        """
        Save scenario files to directory

        dirpath: Path object
        """
        # create target directory
        dirpath.mkdir(parents=True, exist_ok=True)

        # save meta information
        dirpath.joinpath('_meta.inf').write_text(json.dumps({"rng_seed": self.rng_seed}, indent=2))

        # save power network
        dirpath.joinpath('network.cfg').write_text(
            json.dumps(
                { self.power_network.name: self.power_network.to_dict() },
                indent=2,
            )
        )

        # save actors
        for actor in self.actors:
            dirpath.joinpath(f'actor_{actor.id}.cfg').write_text(json.dumps(actor.to_dict(), indent=2))

        # save map_actors
        dirpath.joinpath('map_actors.cfg').write_text(json.dumps(self.map_actors, indent=2))


def from_dict(scenario_dict):
    pn_name, pn_dict = scenario_dict["power_network"].popitem()
    assert len(scenario_dict["power_network"]) == 0, "Multiple power networks in scenario"
    network = json_graph.node_link_graph(pn_dict,
        directed = pn_dict.get("directed", False),
        multigraph = pn_dict.get("multigraph", False))
    pn = power_network.PowerNetwork(pn_name, network)

    actors = [
        actor.Actor(actor_id, pd.read_json(ai["df"]), ai["ls"], ai["ps"], ai["pm"])
        for actor_id, ai in scenario_dict["actors"].items()]

    return Scenario(pn, actors, scenario_dict["map_actors"], scenario_dict["rng_seed"])


def load(dirpath):
    """
    Create scenario from files that were generated by Scenario.save()

    dirpath: Path object
    """

    # read meta info
    meta_text = dirpath.joinpath('_meta.inf').read_text()
    meta = json.loads(meta_text)
    rng_seed = meta.get("rng_seed", None)

    # read power network
    network_text = dirpath.joinpath('network.cfg').read_text()
    network_json = json.loads(network_text)
    network_name = list(network_json.keys())[0]
    network_json = list(network_json.values())[0]
    network = json_graph.node_link_graph(network_json,
        directed = network_json.get("directed", False),
        multigraph = network_json.get("multigraph", False))
    pn = power_network.PowerNetwork(network_name, network)

    # read actors
    actor_files = dirpath.glob("actor_*.cfg")
    actors = []
    for f in sorted(actor_files):
        at = f.read_text()
        aj = json.loads(at)
        ai = [aj["id"], pd.read_json(aj["df"]), aj["ls"], aj["ps"], aj["pm"]]
        actors.append(actor.Actor(*ai))

    # read map_actors
    map_actor_text = dirpath.joinpath('map_actors.cfg').read_text()
    map_actors = json.loads(map_actor_text)

    return Scenario(pn, actors, map_actors, rng_seed)


def create_random(num_nodes, num_actors):
    assert num_actors < num_nodes
    pn = power_network.create_random(num_nodes)
    actors = [actor.create_random(i) for i in range(num_actors)]
    # actors = create_households_from_csv('..\data\households', num_actors)
    # Add actor nodes at random position in the network
    # One network node can contain several actors (using random.choices method)
    map_actors = pn.add_actors_random(actors)

    return Scenario(pn, actors, map_actors)


def create_random2(num_nodes, num_actors):
    assert num_actors < num_nodes
    # num_actors has to be much smaller than num_nodes
    pn = power_network.create_random(num_nodes)
    actors = [actor.create_random(i) for i in range(num_actors)]

    # Give actors a random position in the network
    actor_nodes = random.sample(pn.leaf_nodes, num_actors)
    map_actors = {actor.id: node_id for actor, node_id in zip(actors, actor_nodes)}

    # TODO tbd if actors are already part of topology ore create additional nodes
    # pn.add_actors_map(map_actors)

    return Scenario(pn, actors, map_actors)


def create_households_from_csv(dirpath, num_nodes, num_actors):
    """
    load time series from csv files, source: https://www.loadprofilegenerator.de/results2/
    - 62 sample results from loadprofilegenerator
	- the households are described in household_data_description.csv
	- year is 2016
	- unit is [kWh]
	- original data is in 1 min resolution
    
    """
    # create random nodes for power network
    pn = power_network.create_random(num_nodes)
    
    # create list with alle files in dir '..\data\households'
    filenames = glob.glob(os.path.join(dirpath, "*.csv"))

    # read each load curve in separate DataFrame
    # create list to track data input for each actor
    household_type = []
    # create list of actors
    actors = []
    # choose a random sample of files to read
    filenames = random.sample(filenames, num_actors)
    # iterate over list of files to be read
    for i,filename in enumerate(filenames):
        # save actor_id and data description in list 
        # TODO! where to save it?
        household_type.append((i, os.path.basename(filename)[:-4]))
        print('actor_id: {} - household: {}'.format(i, os.path.basename(filename)[:-4]))
        # read file
        df = pd.read_csv(filename,
                         sep = ';',
                         parse_dates = ['Time'],
                         dayfirst = True)
        df = df.set_index('Time')
        df = df.rename(columns={'Sum [kWh]' : "load"})
        # TODO! The "unsampled" data will have another column called 'Electricity.Timestep'
        # df = df.drop(columns='Electricity.Timestep')
        # TODO! we need realistic data for the PV loads
        df['pv'] = [1]*len(df)
        df['prices'] = [1]*len(df)
        
        actors.append(actor.Actor(i, df))   
    
    map_actors = pn.add_actors_random(actors)
    
    return Scenario(pn, actors, map_actors)