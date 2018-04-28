##########################################################################################
# Parse JSON config file
##########################################################################################
import json
from bunch import Bunch
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # Parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # Convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(args):
    json_file = args.config_file
    config, _ = get_config_from_json(json_file)

    config.mode = args.mode
    config.config_file = args.config_file
    config.summary_dir = os.path.join(config.output_path, config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join(config.output_path, config.exp_name, "checkpoints/")

    return config

