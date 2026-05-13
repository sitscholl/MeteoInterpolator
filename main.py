import logging
from datetime import datetime
import argparse
import asyncio

from src.validate.date import localize_datetime_string
from src.runtime import RuntimeContext
from src.workflow import InterpolationWorkflow

logger = logging.getLogger(__name__)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param', required=True, help = 'Parameter to interpolate, for instance tair_2m')
    parser.add_argument('-s', '--start', required=True, help = 'Start date for interpolation in ISO format, e.g. 2026-05-13')
    parser.add_argument('-e', '--end', required=True, help = 'Exclusive end date for interpolation in ISO format, e.g. 2026-05-15')
    parser.add_argument('-c', '--config', default='config.yaml', help = 'Path to configuration file')
    parser.add_argument('-v', action='store_true', help = 'Turn on verbose logging')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.v else logging.INFO
    logging.basicConfig(level = log_level, force = True, format='%(name)s - %(levelname)s - %(message)s')

    if args.param != 'tair_2m':
        raise NotImplementedError(f"Interpolation is currently only implemented for parameter 'tair_2m'. Got {args.param}")

    logger.info("="*50)
    logger.info("Starting interpolation at %s", datetime.now().strftime("%H:%M:%S"))
    logger.info("="*50)

    runtime = RuntimeContext.from_config_file(args.config)
    start = localize_datetime_string(args.start, runtime.timezone)
    end = localize_datetime_string(args.end, runtime.timezone)
    interpolation_workflow = InterpolationWorkflow(runtime)
    
    logger.info("Initialized Runtime Context and Interpolation Workflow")

    try:
        # runtime.cluster_manager.spin_up_workers() #will be implemented later

        logger.info(f"Starting interpolation workflow for parameter {args.param} over period {start} - {end}")
        result = asyncio.run(interpolation_workflow.run(param = args.param, start = start, end = end))
        logger.info("Interpolation workflow produced %s grid(s)", len(result))

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
