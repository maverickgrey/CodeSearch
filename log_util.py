import logging
import os
import time
from config_class import LOGS_DIR
from pathlib import Path

"""
日志工具文件
"""

class LogUtil:
    def __init__(self,is_console=True,is_file=True,stream_level=logging.INFO,file_level=logging.INFO):
        self.is_console = is_console
        self.is_file = is_file
        self.stream_level = stream_level
        self.file_level = file_level
    
    # 会自动生成日志和日志存储文件的名字，它是按照年月日来命名的
    def save_logs(self):
        path = Path(LOGS_DIR)
        path.mkdir(parents=True, exist_ok=True)
        path = path / '{}.log'.format(time.strftime("%Y%m%d", time.localtime()))
        return str(path)

    # 能够调用该函数获得一个logger，默认情况该logger会向控制台和日志文件中都输出日志
    def get_logger(self,is_console=True,is_file=True):
        log_name = self.save_logs()
        # getLogger中要传入记录log的文件名
        logger = logging.getLogger(log_name)
        # 需要设置好log的级别
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        # stream_handler：控制台输出
        if is_console:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(self.stream_level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
        # file_handler：文件输出
        if is_file:
            file_handler = logging.FileHandler(log_name)
            file_handler.setLevel(self.file_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger