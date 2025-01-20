"""
Logger Configuration for Vessel Motion Monitoring System (VMMS)

This module provides logging configuration utilities for the VMMS system, including
compressed log rotation and flexible logger setup.

Key Components:
- CompressedRotatingFileHandler: Custom handler that compresses rotated log files
  - Inherits from RotatingFileHandler
  - Automatically compresses old log files using gzip
  
- setup_logger(): Creates and configures a logger with both file and console output
  - Supports log rotation with compression
  - Configurable log levels, max file size and backup count
  - Formats logs with timestamps and log levels

Usage Example:
    logger = setup_logger(
        'myapp',
        'app.log',
        level=logging.INFO,
        max_bytes=10*1024*1024,  # 10MB
        backup_count=5
    )
    logger.info("Application started")

Project: Vessel Intelligence Operation System (VIOS)
         Vessel Motion Monitoring System (VMMS)
Developer: Dr. GUO, XIAOXIAN @ SJTU/SKLOE
Contact: xiaoxguo@sjtu.edu.cn

Dependencies:
- logging
- gzip
- os

Copyright (c) 2024 Shanghai Jiao Tong University
All rights reserved.
Date: 2025-01-07
"""

import logging
from logging.handlers import RotatingFileHandler
import gzip
import os

class CompressedRotatingFileHandler(RotatingFileHandler):
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=0):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.rotation_filename("%s.%d.gz" % (self.baseFilename, i))
                dfn = self.rotation_filename("%s.%d.gz" % (self.baseFilename, i + 1))
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            dfn = self.rotation_filename(self.baseFilename + ".1.gz")
            if os.path.exists(dfn):
                os.remove(dfn)
            with open(self.baseFilename, 'rb') as f_in:
                with gzip.open(dfn, 'wb') as f_out:
                    f_out.writelines(f_in)
        if not self.delay:
            self.stream = self._open()

def setup_logger(name, log_file, level=logging.INFO, max_bytes=10*1024*1024, backup_count=5):
    """Function to setup a flexible logger"""

    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 创建 CompressedRotatingFileHandler
    file_handler = CompressedRotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 创建 StreamHandler 用于控制台输出
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # 添加 handlers 到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 设置 Prophet 日志
def setup_prophet_logger(level=logging.ERROR):
    prophet_logger = logging.getLogger('prophet')
    prophet_logger.setLevel(level)

# 可选：设置全局日志级别
def set_global_log_level(level):
    logging.getLogger().setLevel(level)