#!/usr/bin/env python3
from pathlib import Path
from argparse import ArgumentParser

from simply import market, market_2pac, market_fair
from simply.scenario import load, create_random, Scenario
from simply.config import Config
from simply.util import summerize_actor_trading
import os
import glob
"""
Entry point for standalone functionality.

Reads in configuration file (or uses defaults when none is supplied),
creates or loads scenario and matches orders in each timestep.
May show network plots or print market statistics, depending on config.

Usage: python match_market.py [config file]
"""


def main(cfg: Config):

    print(cfg)
    # Checks if actor files with the correct format exist in the cfg.scenario_path
    # --------------------------------
    def list_files_in_path(current_path, pattern='*'):
        file_list = glob.glob(os.path.join(current_path, pattern))
        return file_list

    current_path = Path.cwd()
    print("Hier bin ich, jetzt:", current_path)
    print("Das ist data_format: ", cfg.data_format)
    print(f'Scenario path: {cfg.scenario_path}')
    files_in_path = list_files_in_path(cfg.scenario_path)
    #print(f"${cfg.scenario_path}/*")
    #names = [os.path.basename(x) for x in glob.glob(f"${cfg.scenario_path}/*")]
    print("Das sind die Files in: ", cfg.scenario_path, ":  ", files_in_path)
    # --------------------------------

    scenario_exists = len([False for i in cfg.scenario_path.glob(f"*actor*_*.{cfg.data_format}")]) != 0
    print("scenario_exists: ", scenario_exists)

    # load existing scenario or else create randomized new one
    sc: Scenario

    if cfg.load_scenario:
        print("cfg.scenario_path: ", cfg.scenario_path)
        if scenario_exists:
            sc = load(cfg.scenario_path, cfg.data_format)
        else:
            raise Exception(f'Could not find scenario path: {cfg.scenario_path}. Make sure to include the '
                            f'scenario directory in your project or if you want to generate a random scenario, '
                            f'set load_scenario = False in config.txt.')
    else:
        # Todo: is secnario_path the right path here?
        if cfg.scenario_path.exists():
            raise Exception(f'The path: {cfg.scenario_path} already exists with another file structure. '
                            'Please remove or rename folder to avoid confusion and restart '
                            'simulation.')
        else:
            # create path if it does not exist yet
            # Todo: is secnario_path the right path here?
            cfg.scenario_path.mkdir(parents=True, exist_ok=True)
        sc = create_random(cfg.nb_nodes, cfg.nb_actors, cfg.weight_factor)
        sc.save(cfg.scenario_path, cfg.data_format)

    if cfg.show_plots:
        sc.power_network.plot()
        sc.plot_participant_data()

    # generate requested market
    if "pac" in cfg.market_type:
        m = market_2pac.TwoSidedPayAsClear(network=sc.power_network)
    elif cfg.market_type in ["fair", "merit"]:
        m = market_fair.BestMarket(network=sc.power_network)
    else:
        # default
        m = market.Market(network=sc.power_network)

    sc.add_market(m)
    for _ in range(cfg.nb_ts):
        # actors calculate strategy based market interaction with the market maker
        sc.create_strategies()

        # orders are generated based on the flexibility towards the planned market interaction
        # and a pricing scheme. Orders are matched at the end
        sc.market_step()

        # actors are prepared for the next time step by changing socs, banks and predictions
        sc.next_time_step()

        if cfg.show_prints:
            print("Matches of bid/ask ids: {}".format(m.matches))
            print(
                "\nCheck individual traded energy blocks (splitted) and price at market level"
            )

    if cfg.show_prints:
        print("\nTraded energy volume and price at actor level")
        print(summerize_actor_trading(sc))

    return sc


if __name__ == "__main__":
    parser = ArgumentParser(description='Entry point for market simulation')
    # parser.add_argument('config', nargs='?', default="", help='configuration file')
    # Replaced the above line to take in the project directory (which will contain the config file) instead of putting in the config file
    # also made it mandatory
    parser.add_argument('project_dir', nargs='?', default=None, help='project directory path')
    args = parser.parse_args()
    # Raise error if project directory not specified
    if args.project_dir is None:
        raise (
            FileNotFoundError(
                "Project directory path must be specified. Please provide the path as a command-line argument."))
    # This means that the config file must always be in the project directory
    config_file = os.path.join(args.project_dir, "config.txt")
    # Raise error if config.txt file not found in project directory
    if not os.path.isfile(config_file):
        raise (
            FileNotFoundError(f"Config file not found in project directory: {args.project_dir}"))
    cfg = Config(config_file, args.project_dir)
    main(cfg)