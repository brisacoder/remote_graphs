# Description: This file contains a sample graph client that makes a stateless request to the Remote Graph Server.
# Usage: python3 client/rest.py

import json
import traceback
import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

import requests
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import Command
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages.utils import convert_to_openai_messages
from logging_config import configure_logging
from requests.exceptions import (
    ConnectionError as RequestsConnectionError,
    HTTPError,
    RequestException,
    Timeout,
)

# Initialize logger
logger = configure_logging()

# URL for the Remote Graph Server /runs endpoint
REMOTE_SERVER_URL = "http://127.0.0.1:8123/api/v1/runs"


def load_environment_variables(env_file: str | None = None) -> None:
    """
    Load environment variables from a .env file safely.

    This function loads environment variables from a `.env` file, ensuring
    that critical configurations are set before the application starts.

    Args:
        env_file (str | None): Path to a specific `.env` file. If None,
                               it searches for a `.env` file automatically.

    Behavior:
    - If `env_file` is provided, it loads the specified file.
    - If `env_file` is not provided, it attempts to locate a `.env` file in the project directory.
    - Logs a warning if no `.env` file is found.

    Returns:
        None
    """
    env_path = env_file or find_dotenv()

    if env_path:
        load_dotenv(env_path, override=True)
        logger.info(f".env file loaded from {env_path}")
    else:
        logger.warning("No .env file found. Ensure environment variables are set.")


def decode_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decodes the JSON response from the remote server and extracts relevant information.

    Args:
        response_data (Dict[str, Any]): The JSON response from the server.

    Returns:
        Dict[str, Any]: A structured dictionary containing extracted response fields.
    """
    try:
        agent_id = response_data.get("agent_id", "Unknown")
        output = response_data.get("output", {})
        model = response_data.get("model", "Unknown")
        metadata = response_data.get("metadata", {})

        # Extract messages if present
        messages = output.get("messages", [])

        return {
            "agent_id": agent_id,
            "messages": messages,
            "model": model,
            "metadata": metadata,
        }
    except Exception as e:
        return {"error": f"Failed to decode response: {str(e)}"}


# Define the graph state
class GraphState(TypedDict):
    """Represents the state of the graph, containing a list of messages."""

    messages: Annotated[List[BaseMessage], add_messages]
    exception_msg: str


def default_state() -> Dict:
    """
    A benign default return for nodes in the graph
    that do not modify state
    """
    return {
        "messages": [],
    }


# Graph node that makes a stateless request to the Remote Graph Server
def node_remote_agent(
    state: GraphState,
) -> Command[Literal["exception_node", "end_node"]]:
    """
    Sends a stateless request to the Remote Graph Server.

    Args:
        state (GraphState): The current graph state containing messages.

    Returns:
        Dict[str, List[BaseMessage]]: Updated state containing server response or error message.
    """
    if not state["messages"]:
        logger.error(json.dumps({"error": "GraphState contains no messages"}))
        return Command(
            goto="exception_node",
            update={"exception_text": "Error: No messages in state"},
        )

    # Extract the latest user query
    human_message = state["messages"][-1].content
    logger.info(json.dumps({"event": "sending_request", "human": human_message}))

    # Request headers
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    messages = convert_to_openai_messages(state["messages"])

    # payload to send to autogen server at /runs endpoint
    payload = {
        "agent_id": "remote_agent",
        "input": {"messages": messages},
        "model": "gpt-4o",
        "metadata": {"id": str(uuid.uuid4())},
    }

    # Use a session for efficiency
    session = requests.Session()

    try:
        response = session.post(
            REMOTE_SERVER_URL, headers=headers, json=payload, timeout=30
        )

        # Raise exception for HTTP errors
        response.raise_for_status()

        # Parse response as JSON
        response_data = response.json()
        # Decode JSON response
        decoded_response = decode_response(response_data)

        logger.info(decoded_response)

        messages = decoded_response.get("messages", [])

        # This is tricky. In multi-turn conversation we should only add new messages
        # produced by the remote agent, otherwise we will have duplicates.
        # In this App we will assume remote agent only create a single new message but
        # this is not always true

        return Command(goto="end_node", update={"messages": messages[-1]})

    except (Timeout, RequestsConnectionError) as conn_err:
        error_msg = {
            "error": "Connection timeout or failure",
            "exception": str(conn_err),
        }
        logger.error(json.dumps(error_msg))

        return Command(
            goto="exception_node", update={"exception_msg": json.dumps(error_msg)}
        )

    except HTTPError as http_err:
        error_msg = {
            "error": "HTTP request failed",
            "status_code": response.status_code,
            "exception": str(http_err),
        }
        logger.error(json.dumps(error_msg))
        return Command(
            goto="exception_node", update={"exception_msg": json.dumps(error_msg)}
        )

    except RequestException as req_err:
        error_msg = {"error": "Request failed", "exception": str(req_err)}
        logger.error(json.dumps(error_msg))
        return Command(
            goto="exception_node", update={"exception_msg": json.dumps(error_msg)}
        )

    except json.JSONDecodeError as json_err:
        error_msg = {"error": "Invalid JSON response", "exception": str(json_err)}
        logger.error(json.dumps(error_msg))
        return Command(
            goto="exception_node", update={"exception_msg": json.dumps(error_msg)}
        )

    except Exception as e:
        error_msg = {
            "error": "Unexpected failure",
            "exception": str(e),
            "stack_trace": traceback.format_exc(),
        }
        logger.error(json.dumps(error_msg))
        return Command(
            goto="exception_node", update={"exception_msg": json.dumps(error_msg)}
        )

    finally:
        session.close()


# Graph node that makes a stateless request to the Remote Graph Server
def end_node(state: GraphState) -> Dict[str, Any]:
    logger.info(f"Thread end: {state.values()}")
    return default_state()


def exception_node(state: GraphState):
    logger.info(f"Exception happen while processing graph: {state["exception_msg"]}")
    return default_state()


# Build the state graph
def build_graph() -> Any:
    """
    Constructs the state graph for handling requests.

    Returns:
        StateGraph: A compiled LangGraph state graph.
    """
    builder = StateGraph(GraphState)
    builder.add_node("node_remote_agent", node_remote_agent)
    builder.add_node("end_node", end_node)
    builder.add_node("exception_node", exception_node)

    builder.add_edge(START, "node_remote_agent")
    builder.add_edge("exception_node", END)
    builder.add_edge("end_node", END)
    return builder.compile()


def invoke_graph(
    messages: List[Dict[str, str]], graph: Optional[Any] = None
) -> Optional[dict[Any, Any] | list[dict[Any, Any]]]:
    """
    Invokes the graph with the given messages and safely extracts the last AI-generated message.

    - Logs errors if keys or indices are missing.
    - Ensures the graph is initialized if not provided.
    - Returns a meaningful response even if an error occurs.

    :param messages: A list of message dictionaries in OpenAI format
    :param graph: An optional graph object to use; internal will be built if not provided.
    :return: The list of all messages returned by the graph
    """
    inputs = {"messages": messages}
    logger.debug({"event": "invoking_graph", "inputs": inputs})

    try:
        if not graph:
            graph = build_graph()

        result = graph.invoke(inputs)

        if not isinstance(result, dict):
            raise TypeError(
                f"Graph invocation returned non-dict result: {type(result)}"
            )

        messages_list = convert_to_openai_messages(result.get("messages", []))
        if not isinstance(messages_list, list) or not messages_list:
            raise ValueError("Graph result does not contain a valid 'messages' list.")

        last_message = messages_list[-1]
        if not isinstance(last_message, dict) or "content" not in last_message:
            raise KeyError(f"Last message does not contain 'content': {last_message}")

        ai_message_content = last_message["content"]
        logger.info(f"AI message content: {ai_message_content}")
        return messages_list

    except Exception as e:
        logger.error(f"Error invoking graph: {e}", exc_info=True)
        return [{"role": "assistant", "content": "Error processing user message"}]


def main():
    graph = build_graph()
    inputs = {"messages": [HumanMessage(content="Write a story about a cat")]}
    logger.info({"event": "invoking_graph", "inputs": inputs})
    result = graph.invoke(inputs)
    logger.info({"event": "final_result", "result": result})


# Main execution
if __name__ == "__main__":
    main()
