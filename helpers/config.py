# -*- coding: utf-8 -*-
"""
Helper function to parse config files (without section headers).
"""

import configparser


CONFIG_PATH = 'config.cfg'


def get(key):
    try:
        with open(CONFIG_PATH, 'r') as f:
            config_string = '[DEFAULT]\n' + f.read()
        config = configparser.ConfigParser()
        config.read_string(config_string)
        return config['DEFAULT'][key]
    except: return False
    return False
