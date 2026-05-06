import sys
import logging
from pathlib import Path

# Bootstrap logging before any project imports
sys.path.insert(0, str(Path(__file__).parent))
from src.logger import get_logger, LOG_FILE

logger = get_logger(__name__)

from src.pipeline import run_full_pipeline

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Starting AI Financial Behavior Project pipeline...")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("=" * 80)

    try:
        outputs = run_full_pipeline()

        logger.info("\n" + "=" * 80)
        logger.info("Pipeline completed successfully.")
        logger.info("Returned output keys:")
        if isinstance(outputs, dict):
            logger.info(str(list(outputs.keys())))
        else:
            logger.info(str(outputs))
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)