"""Main entry point for P&ID FROM-TO Agent."""

import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

from src.agents.pid_agent import PIDAgent
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """
    Main entry point.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="P&ID FROM-TO Agent - Identify pipe connections in P&ID diagrams"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the P&ID image",
    )
    parser.add_argument(
        "pipe_label",
        type=str,
        help="Target pipe label (e.g., P-101)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results",
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Validate input
    if not Path(args.image_path).exists():
        logger.error(f"Image not found: {args.image_path}")
        return 1

    try:
        # Initialize agent
        logger.info("Initializing P&ID Agent...")
        agent = PIDAgent(model_name=args.model)

        # Identify pipe
        logger.info(f"Identifying pipe: {args.pipe_label}")
        result = agent.identify_pipe_from_to(
            image_path=args.image_path,
            target_pipe=args.pipe_label,
        )

        # Print result
        print("\n" + "="*50)
        print("P&ID FROM-TO Analysis Result")
        print("="*50)
        print(json.dumps(result, indent=2))
        print("="*50 + "\n")

        # Save to file if specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"Result saved to: {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
