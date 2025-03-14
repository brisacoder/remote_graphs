"""
This module provides functions to build and invoke graphs using the AP REST client.

Functions:
- build_local_graph: Builds a local graph using the AP protocol.
"""

from logging_config import configure_logging
from ap_rest_client.ap_protocol import invoke_graph  # type: ignore

# Initialize logger
logger = configure_logging()


messages = [{"role": "user", "content": "Write a story about a cat"}]
print(invoke_graph(messages=messages))
