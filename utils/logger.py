import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir='logs', log_file=None, name='default', level=logging.INFO):
        """
        :param log_dir: 日志保存目录
        :param log_file: 日志文件名，不提供则自动生成
        :param name: logger 名称
        :param level: logging level，比如 logging.INFO
        """
        os.makedirs(log_dir, exist_ok=True)

        if log_file is None:
            log_file = datetime.now().strftime('%Y%m%d_%H%M%S') + '.log'
        log_path = os.path.join(log_dir, log_file)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # 防止重复打印

        # 防止重复添加 handler
        if not self.logger.handlers:
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

            # 控制台输出
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # 文件输出
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def debug(self, msg):
        self.logger.debug(msg)

    def critical(self, msg):
        self.logger.critical(msg)
