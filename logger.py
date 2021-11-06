import logging
import os
import torch.distributed as dist

_FORMAT = '%(message)s'
logging.basicConfig(level=logging.INFO, format=_FORMAT)


class DistributedLogger:
    def __init__(self, name, level='INFO', log_dir: str = None, mode='a'):
        assert dist.is_initialized(
        ), 'DistributedLogger should be initialized after dist.is_initialized()'
        if dist.get_rank() == 0:
            self._logger = logging.getLogger(name)
            self._logger.setLevel(getattr(logging, level))

            if log_dir is not None:
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, f'{name}.log')
                file_handler = logging.FileHandler(log_path, mode)
                file_handler.setLevel(getattr(logging, level))
                formatter = logging.Formatter(_FORMAT)
                file_handler.setFormatter(formatter)
                self._logger.addHandler(file_handler)

    def info(self, message: str):
        if dist.get_rank() == 0:
            self._logger.info(message)

    def warning(self, message: str):
        if dist.get_rank() == 0:
            self._logger.warning(message)

    def debug(self, message: str):
        if dist.get_rank() == 0:
            self._logger.debug(message)

    def error(self, message: str):
        if dist.get_rank() == 0:
            self._logger.error(message)
