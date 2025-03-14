from logging_config import configure_logging
from ap_rest_client.ap_protocol import build_graph, invoke_graph

# Initialize logger
logger = configure_logging()


def build_local_graph():
    return build_graph()


# messages = [{"role": "user", "content": "Write a story about a cat"}]
# invoke_graph(messages=messages)
