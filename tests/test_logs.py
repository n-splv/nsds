import logging

from nsds.logs import configure_logging


def test_configure_logging_sets_up_the_notebook_logger():
    configure_logging()

    logger = logging.getLogger("notebook")
    assert logger.level == logging.DEBUG
    assert logger.propagate is False
    assert len(logger.handlers) == 1
