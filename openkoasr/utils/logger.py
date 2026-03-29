# Copyright (c) 2025 Gwantae Kim. All Rights Reserved.
# Licensed under the MIT License.

import logging
import os
from datetime import datetime

class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, name: str = 'ASR_Logger'):
        if hasattr(self, 'logger'):  # Avoid re-initialization
            return

        log_dir = "log"
        os.makedirs(log_dir, exist_ok=True)

        log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

            # File handler
            file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Stream handler
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

    def get_logger(self):
        return self.logger

# Instantiate the logger so it can be imported directly
logger = Logger().get_logger()
