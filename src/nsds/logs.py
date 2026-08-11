import logging.config

CONFIG = {
    'version': 1,
    'formatters': {
        'default': {'format': '%(asctime)s - %(levelname)s - %(message)s'},
    },
    'handlers': {
        'stream': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        },
    },
    'loggers': {
        'notebook': {
            'handlers': ['stream'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}


def configure_logging() -> None:
    logging.config.dictConfig(CONFIG)
