"""
Logging configuration and utilities for the ChartRadar framework.

This module provides structured logging setup with file and console handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "chartradar",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure a logger for the ChartRadar framework.
    
    Args:
        name: Logger name (default: "chartradar")
        level: Logging level (default: INFO)
        log_file: Optional path to log file (if None, only console logging)
        format_string: Optional custom format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Default format: timestamp, level, name, message
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance, creating it if it doesn't exist.
    
    Args:
        name: Optional logger name (default: "chartradar")
        
    Returns:
        Logger instance
    """
    logger_name = name or "chartradar"
    logger = logging.getLogger(logger_name)
    
    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        setup_logger(logger_name)
    
    return logger

