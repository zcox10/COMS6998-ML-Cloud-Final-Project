import torch
import time
import os
import re
import numpy as np
import logging


class GenericUtils:
    def configure_component_logging(self, log_level):
        """
        Configure root logger for component logs.
        """
        logger = logging.getLogger()  # Get the root logger
        logger.setLevel(log_level)

        # Clear existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        # Add console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def set_seed(self, seed):
        """
        Sets a random seed throughout the notebook.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    def set_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    def time_operation(self, start, message):
        end = time.perf_counter()
        elapsed = end - start
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        logging.info(f"{message}: {minutes} min {seconds:.2f} sec")
        return round(elapsed / 60, 2)

    def create_dirs(self, dirs_to_create):
        # Create directories if they don't exist
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            if os.path.isdir(dir_path):
                logging.info(f"Created or already exists: {dir_path}")
            else:
                logging.info(f"Failed to create: {dir_path}")

    def _extract_part_timestamp(self, filename):
        pattern = r"^(part_\d+_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})"
        match = re.match(pattern, filename)
        return match.group(1) if match else None
