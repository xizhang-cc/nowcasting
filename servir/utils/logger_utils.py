import os
import time
import logging

# from servir.utils.main_utils import collect_env
from .main_utils import collect_env

def logging_setup(log_path, fname='log.log', log_env_info=True):
    """Set up logger to save the running log.
    Args:
        log_path_path (str): Path to save the log.
    """

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    fname_wt = fname.rsplit('.', 1)[0] + '_' + timestamp + '.log'

    logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(log_path, fname_wt),
                        filemode='a', format='%(asctime)s - %(message)s')

    # log env info
    dash_line = '-' * 60 + '\n'

    if log_env_info:
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    
        logging.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)


def logging_config_info(config):
    # log config
    if config is not None:
        dash_line = '-' * 60 + '\n'
        config_info = '\n'.join([(f'{k}: {v}') for k, v in config.items()])
        logging.info('Config Info:\n' + dash_line + config_info + '\n' + dash_line)

def logging_method_info(method, device):
    pass