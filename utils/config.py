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

    config_dict_mod = {}
    for name, value in config_dict.items():
        # Ignore comments
        if name[0] != '_':
            config_dict_mod[name] = value


    # Convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict_mod)

    return config, config_dict_mod


def process_config(args):
    json_file = args.config_file
    config, _ = get_config_from_json(json_file)

    config.mode = args.mode
    config.config_file = args.config_file
    config.summary_dir = os.path.join(config.output_path, config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join(config.output_path, config.exp_name, "checkpoints/")
    config.image_shape = (config.tfr_image_height, config.tfr_image_width, config.tfr_image_channels)

    return config

