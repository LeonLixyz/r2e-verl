import logging
import sys


def setup_logging(
    level: int = logging.INFO,
    extra_modules: list[str] | None = None,
) -> None:
    modules = ["relace_agent"]
    if extra_modules:
        modules.extend(extra_modules)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    for module in modules:
        logger = logging.getLogger(module)
        logger.setLevel(level)
        logger.addHandler(stream_handler)
