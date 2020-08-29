#suppress verbose warnings
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

__all__ = ["models", "pipeline", "evaluate"]