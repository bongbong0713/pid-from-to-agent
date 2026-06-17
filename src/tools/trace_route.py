"""Tool for tracing pipe routes in P&ID diagrams."""

from typing import Dict, Any
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def trace_route(
    graph,
    pipe_label: str,
) -> Dict[str, Any]:
    """
    Trace the route of a specific pipe.

    Args:
        graph: NetworkX graph object
        pipe_label: Label of the pipe to trace

        Returns:
        Dictionary with from/to equipment and path details
    """
    try:
        from src.graph.tracer import PIDTracer

        tracer = PIDTracer(graph)
        result = tracer.trace_pipe_connections(pipe_label)

        logger.info(f"Traced route for pipe: {pipe_label}")
        return result
    except Exception as e:
        logger.error(f"Error tracing route: {e}")
        return {
            "error": str(e),
            "pipe_label": pipe_label,
        }
