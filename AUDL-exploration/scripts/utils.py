import sqlite3
import yaml
import sys
import logging

from sqlite3 import Error


def create_connection(db_file):
    """ 
    create a database connection to the SQLite database specified by db_file

    Parameters
    ----------
    db_file: string
        path to database (ex: *.db)
        
    Returns
    -------
    connection: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        logging.error(e)
        sys.exit(1)

    return conn

def load_config(config_file):
    """ 
    Load config file

    Parameters
    ----------
    config_file: yaml (ex: *.cfg)
        path to config file

    Returns
    -------
    yaml_dict: dict
        dict with yaml configs
    
    """
    try: 
        with open(config_file) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            return data
    except OSError as e:
        logging.error(e)
        sys.exit(1)
    
