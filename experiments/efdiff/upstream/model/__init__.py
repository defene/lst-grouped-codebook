import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .model import DDPM as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(opt['model'].get('which_model_G', 'unknown')))
    return m
