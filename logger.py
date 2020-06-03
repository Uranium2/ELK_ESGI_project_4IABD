import logging

def logger_init():
    # create logger
    logger = logging.getLogger('AWS_project')
    logger.setLevel(logging.DEBUG)

    # create console handler & file handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    fh = logging.FileHandler('AWS_project.log')
    fh.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
