import logging

import structlog


def configure_logging(debug: bool) -> None:
    """
    Configure structlog for CLI usage.

    This configures a console-friendly renderer, timestamper and exception helpers,
    and sets the minimum log level according to `debug`.
    """
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.set_exc_info,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            min_level=logging.DEBUG if debug else logging.INFO
        ),
        cache_logger_on_first_use=True,
    )
