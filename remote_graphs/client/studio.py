"""
This module provides the entry point for Langgraph Studio to build a local graph.
"""

from typing import Any

from ap_rest_client.ap_protocol import build_graph
from logging_config import configure_logging

# Initialize logger
logger = configure_logging()


def build_local_graph() -> Any:
    """
    Entry point for Langgraph Studio. Builds a local graph.

    Returns:
        The built graph object.
    """
    return build_graph()
