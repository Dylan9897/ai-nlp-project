import os
import datetime
import logging

log_path = "log"
log_name = "{}.log".format(datetime.datetime.now().strftime('%Y-%m-%d'))

if not os.path.exists(log_path):
    os.makedirs(log_path)

logger=logging.getLogger("client_log")

logger.setLevel(logging.INFO)

stream_handler=logging.StreamHandler()
log_file_handler=logging.FileHandler(filename=os.path.join(log_path,log_name),encoding="utf-8")

formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(message)s")

stream_handler.setFormatter(formatter)
log_file_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(log_file_handler)