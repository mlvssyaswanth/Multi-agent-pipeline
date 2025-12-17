"""
Enhanced logging configuration for Multi-Agent Coding Framework.
Provides structured logging with timestamps, context, and performance metrics.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logging(
    level=logging.INFO,
    log_to_file=False,
    log_file_path="logs/multi_agent_framework.log",
    include_timestamp=True,
    include_context=True
):
    """
    Setup enhanced logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_file_path: Path to log file
        include_timestamp: Include timestamp in logs
        include_context: Include context information (module, function, line)
    """
    # Create logs directory if logging to file
    if log_to_file:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatter - minimalistic format
    format_string = '%(asctime)s | %(levelname)-8s | %(message)s'
    date_format = '%H:%M:%S'
    
    formatter = logging.Formatter(format_string, datefmt=date_format)
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(format_string, datefmt=date_format)
    console_handler.setFormatter(console_formatter)
    
    # File handler (plain text, no colors)
    handlers = [console_handler]
    if log_to_file:
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Set specific logger levels
    logging.getLogger('httpx').setLevel(logging.WARNING)  # Reduce httpx verbosity
    logging.getLogger('httpcore').setLevel(logging.WARNING)  # Reduce httpcore verbosity
    logging.getLogger('autogen').setLevel(logging.INFO)  # Control autogen logging
    
    return logging.getLogger(__name__)


def get_logger(name):
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class PerformanceLogger:
    """Context manager for logging performance metrics."""
    
    def __init__(self, logger, operation_name, log_level=logging.INFO):
        """
        Initialize performance logger.
        
        Args:
            logger: Logger instance
            operation_name: Name of the operation being timed
            log_level: Log level for the performance message
        """
        self.logger = logger
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            status = "Completed" if exc_type is None else "Failed"
            self.logger.log(
                self.log_level,
                f"{self.operation_name}: {status} ({duration:.2f}s)"
            )
        return False  # Don't suppress exceptions


def log_api_call(logger, agent_name, model, prompt_length, response_length=None):
    """
    Log API call details.
    
    Args:
        logger: Logger instance
        agent_name: Name of the agent making the call
        model: Model being used
        prompt_length: Length of the prompt
        response_length: Length of the response (if available)
    """
    # Minimal logging - only log if needed for debugging
    pass  # Removed verbose API call logging


def log_agent_activity(logger, agent_name, activity, details=None):
    """
    Log agent activity.
    
    Args:
        logger: Logger instance
        agent_name: Name of the agent
        activity: Description of the activity
        details: Additional details (dict)
    """
    # Minimal logging - only log activity name
    logger.info(f"{agent_name}: {activity}")

