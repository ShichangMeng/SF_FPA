import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch


class AverageMeter:
    """Track the running average of a scalar metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fix_randomness(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _logger(logger_name, level=logging.DEBUG):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(logger_name, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def starting_logs(data_type, da_method, exp_log_dir, src_id, tgt_id, run_id):
    log_dir = os.path.join(exp_log_dir, f"{src_id}_to_{tgt_id}_run_{run_id}")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    logger = _logger(os.path.join(log_dir, f"logs_{timestamp}.log"))
    logger.debug("=" * 45)
    logger.debug(f"Dataset: {data_type}")
    logger.debug(f"Method:  {da_method}")
    logger.debug("=" * 45)
    logger.debug(f"Source: {src_id} ---> Target: {tgt_id}")
    logger.debug(f"Run ID: {run_id}")
    logger.debug("=" * 45)
    return logger, log_dir
