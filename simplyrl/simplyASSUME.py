# Import necessary packages
import pandas as pd
import logging
import os
import yaml
import numpy as np
# import the main World class and the load_scenario_folder functions from assume
from assume import World, load_scenario_folder, run_learning
from learning_strategies_simply import SimplyRLStrategy



def create_scenario_rl(path):
    # Create scenario

    # Set up logging
    log = logging.getLogger(__name__)

    # Create directories if they don't exist
    os.makedirs("local_db", exist_ok=True)
    os.makedirs(path, exist_ok=True)

    # Set the random seed for reproducibility
    np.random.seed(0)

    # Create the units data
    powerplant_units_data = {
        "name": ["Unit 1", "Unit 2", "Unit 3", "Unit 4"],
        "technology": ["pv", "lignite", "hard coal", "combined cycle gas turbine"],
        "bidding_energy": ["prosumer_learning", "naive", "naive", "naive"],
        "fuel_type": ["pv", "lignite", "hard coal", "natural gas"],
        "emission_factor": [0.0, 0.4, 0.3, 0.2],
        "max_power": [15.0, 1000.0, 1000.0, 1000.0],
        "min_power": [0.0, 200.0, 200.0, 200.0],
        "efficiency": [0.3, 0.5, 0.4, 0.6],
        "fixed_cost": [0.3, 1.65, 1.3, 3.5],
        "unit_operator": ["Operator 1", "Operator 2", "Operator 3", "Operator 4"],
    }

    # Convert to DataFrame and save as CSV
    powerplant_units_df = pd.DataFrame(powerplant_units_data)
    powerplant_units_df.to_csv(f"{path}/powerplant_units.csv", index=False)

    # Create the fuel price data
    fuel_prices_data = {
        "fuel": ["pv", "lignite", "hard coal", "natural gas", "oil", "biomass", "co2"],
        "price": [0, 2, 10, 25, 40, 20, 25],
    }

    # Convert to DataFrame and save as CSV
    fuel_prices_df = pd.DataFrame(fuel_prices_data).T
    fuel_prices_df.to_csv(f"{path}/fuel_prices_df.csv", index=True, header=False)

    # Create the demand unit data
    demand_units_data = {
        "name": ["demand_EOM"],
        "technology": ["inflex_demand"],
        "bidding_energy": ["naive"],
        "max_power": [1000000],
        "min_power": [0],
        "unit_operator": ["eom_de"],
    }

    # Convert to DataFrame and save as CSV
    demand_units_df = pd.DataFrame(demand_units_data)
    demand_units_df.to_csv(f"{path}/demand_units.csv", index=False)

    # Create a datetime index for a week with hourly resolution
    date_range = pd.date_range(start="2021-03-01", periods=8 * 24, freq="H")

    # Generate random demand values around 2000
    demand_values = np.random.normal(loc=2000, scale=200, size=8 * 24)

    # Create a DataFrame for the demand profile and save as CSV
    demand_profile = pd.DataFrame({"datetime": date_range, "demand_EOM": demand_values})
    demand_profile.to_csv(f"{path}/demand_df.csv", index=False)


def set_config(path):

    # Define the config as a dictionary
    config_data = {
        "hourly_market": {
            "start_date": "2021-03-01 00:00",
            "end_date": "2021-03-08 00:00",
            "time_step": "1h",
            "save_frequency_hours": None,
            "markets_config": {
                "EOM": {
                    "operator": "EOM_operator",
                    "product_type": "energy",
                    "opening_frequency": "1h",
                    "opening_duration": "1h",
                    "products": [{"duration": "1h", "count": 1, "first_delivery": "1h"}],
                    "volume_unit": "MWh",
                    "price_unit": "EUR/MWh",
                    "market_mechanism": "pay_as_clear",
                }
            },
        }
    }

    # Save the configuration as YAML
    with open(f"{path}/config.yaml", "w") as file:
        yaml.dump(config_data, file, sort_keys=False)


def add_learning_config(path):

    learning_config = {
        "observation_dimension": 50,
        "action_dimension": 2,
        "continue_learning": False,
        "load_model_path": "None",
        "max_bid_price": 100,
        "algorithm": "matd3",
        "learning_rate": 0.001,
        "training_episodes": 50,
        "episodes_collecting_initial_experience": 5,
        "train_freq": 24,
        "gradient_steps": -1,
        "batch_size": 256,
        "gamma": 0.99,
        "device": "cpu",
        "noise_sigma": 0.1,
        "noise_scale": 1,
        "noise_dt": 1,
        "validation_episodes_interval": 5,
    }

    # Read the YAML file
    with open(f"{path}/config.yaml", "r") as file:
        data = yaml.safe_load(file)

    # store our modifications to the config file
    data["hourly_market"]["learning_mode"] = True
    data["hourly_market"]["learning_config"] = learning_config

    # Write the modified data back to the file
    with open(f"{path}/config.yaml", "w") as file:
        yaml.safe_dump(data, file)


if __name__ == "__main__":
    """
    Available examples:
    - local_db: without database and grafana
    - timescale: with database and grafana (note: you need docker installed)
    """
    log = logging.getLogger(__name__)

    data_format = "local_db"  # "local_db" or "timescale"

    if data_format == "local_db":
        db_uri = "sqlite:///./local_db/assume_db.db"
    elif data_format == "timescale":
        db_uri = "postgresql://assume:assume@localhost:5432/assume"

    # define scenario name
    scenario = "simplyASSUME_test"
    # Define paths for input and output data
    csv_path = "./outputs_assume"
    input_path = "./inputs_assume"
    study_case = "hourly_market"
    scenario_path = f"{input_path}/{scenario}"

    # create scenario and set configurations for ASSUME market and learning
    create_scenario_rl(scenario_path)
    set_config(scenario_path)
    add_learning_config(scenario_path)

    # create world
    world = World(database_uri=db_uri, export_csv_path=csv_path)

    # we import our defined bidding strategy class including the learning into the world bidding strategies
    # in the example files we provided the name of the learning bidding strategies in the input csv is  "pp_learning"
    # hence we define this strategy to be one of the learning class
    # world.bidding_strategies["pp_learning"] = RLStrategy
    world.bidding_strategies["prosumer_learning"] = SimplyRLStrategy

    # then we load the scenario specified above from the respective input files
    load_scenario_folder(
        world,
        inputs_path=input_path,
        scenario=scenario,
        study_case=study_case,
    )

    # run learning if learning mode is enabled
    # needed as we simulate the modelling horizon multiple times to train reinforcement learning
    # run_learning( world, inputs_path=input_path, scenario=scenario, study_case=study_case, ...)

    if world.learning_config.get("learning_mode", False):
        run_learning(
            world,
            inputs_path=input_path,
            scenario=scenario,
            study_case=study_case,
        )

    # after the learning is done we make a normal run of the simulation, which equals a test run
    world.run()
