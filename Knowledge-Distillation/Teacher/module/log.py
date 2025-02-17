import re
import logging
from logging.handlers import TimedRotatingFileHandler
import os

# log_dir = os.getenv('logs-base-url')
# pod_name = os.getenv('MY_POD_NAME')

log_dir = 'Teacher/log/'
pod_name = 'test'

def log_init(pod_name, log_dir):
    logging.basicConfig(level=logging.INFO)
    log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
    formatter = logging.Formatter(log_fmt)
    log_file_handler = TimedRotatingFileHandler(filename=log_dir + 'nlp_' + pod_name, when="D", interval=1,
                                                backupCount=7)
    log_file_handler.suffix = "%Y-%m-%d_%H-%M.log"
    log_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.log$")
    log_file_handler.setFormatter(formatter)
    log_file_handler.setLevel(logging.INFO)
    log = logging.getLogger()
    log.addHandler(log_file_handler)
    return log

logger = log_init(pod_name, log_dir)
