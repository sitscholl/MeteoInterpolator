import logging
from datetime import datetime
import argparse

from src.validate.date import localize_datetime_string
from src.constants import NOISY_LOGGERS
from src.runtime import RuntimeContext
from src.workflow import InterpolationWorkflow

logger = logging.getLogger(__name__)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param', help = 'Parameter to interpolate, for instance tair_2m')
    parser.add_argument('-s', '--start', help = 'Start date for interpolation in ISO Format, e.g. 2026-05-13T14:30:00')
    parser.add_argument('-e', '--end', help = 'End date for interpolation in ISO Format, e.g. 2026-05-15T14:30:00')
    parser.add_argument('-c', '--config', default='config.yaml', help = 'Path to configuration file')
    parser.add_argument('-v', action='store_true', help = 'Turn on verbose logging')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.v else logging.INFO
    logging.basicConfig(level = log_level, force = True, format='%(name)s - %(levelname)s - %(message)s')
    if not args.v:
        for logger_name in NOISY_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    if args.param != 'tair_2m':
        raise NotImplementedError(f"Interpolation is currently only implemented for parameter 'tair_2m'. Got {args.param}")

    logger.info("="*50)
    logger.info("Starting interpolation at %s", datetime.now().strftime("%H:%M:%S"))
    logger.info("="*50)

    start, end = localize_datetime_string(args.start), localize_datetime_string(args.end)
    runtime = RuntimeContext.from_config_file(args.config)
    interpolation_workflow = InterpolationWorkflow(runtime)
    
    logger.info("Initialized Runtime Context and Interpolation Workflow")

    try:
        # runtime.cluster_manager.spin_up_workers() #will be implemented later

        logger.info(f"Starting interpolation workflow for parameter {args.param} over period {start} - {end}")
        result = interpolation_workflow.run(param = args.param, start = start, end = end)

    except Exception:
        logger.exception("Error running workflow")
        raise
    finally:
        # runtime.cluster_manager.stop_cluster()

        logger.info("="*50)
        logger.info("Finished interpolation at %s", datetime.now().strftime("%H:%M:%S"))
        logger.info("="*50)


if __name__ == "__main__":
    main()
