import logging

def setup_logger(name, log_file, level=logging.INFO, is_console=True):
    """设置一个日志记录器，输出到指定文件"""
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    if is_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
    return logger

def change_log_file(logger, new_log_file):
    """更改现有日志记录器的输出文件"""
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    new_handler = logging.FileHandler(new_log_file,mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    new_handler.setFormatter(formatter)

    logger.addHandler(new_handler)

    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # logger.addHandler(console_handler)

# # 设置初始日志文件
# logger = setup_logger('my_logger', 'log/original.log')
# logger.info('This log goes to the original file.')
#
# # 更改日志文件
# change_log_file(logger, 'log/new.log')
# logger.info('This log goes to the new file.')
#
