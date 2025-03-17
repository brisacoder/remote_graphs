"""
This module provides functions to build and invoke graphs using the AP REST client.

Functions:
- build_local_graph: Builds a local graph using the AP protocol.
"""

import os
from logging_config import configure_logging
from ap_rest_client.ap_protocol import invoke_graph  # type: ignore
from ap_rest_client.models.graph_config import GraphConfig, RemoteAgentConfig
from dotenv import load_dotenv
# Initialize logger

load_dotenv(override=True)
logger = configure_logging()

remote_agent_url = os.getenv(
    "REMOTE_AGENT_URL", "http://127.0.0.1:8123/api/v1/runs"
)
rest_timeout = int(os.getenv("REST_TIMEOUT", "30"))

graph_config = GraphConfig(
    rest_timeout=rest_timeout,
    # thread_id=str(uuid.uuid4()), If you want to use a specific thread_id, uncomment this line
    remote_agent=RemoteAgentConfig(
        url=remote_agent_url,
        id="remote_agent",
        model="gpt-4o",
        metadata={},
    ),
)


messages = [{"role": "user", "content": "Write a story about a cat"}]
print(invoke_graph(messages=messages, graph_config=graph_config))
