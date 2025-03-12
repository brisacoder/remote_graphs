# Build the state graph

import os
import sys
from typing import Annotated, Any, Dict, List, TypedDict


# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from core.logging_config import configure_logging
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from prompts import Prompts


load_dotenv(override=True)
# Initialize logger
logger = configure_logging()


# Define the graph state
class GraphState(TypedDict):
    """Represents the state of the graph, containing a list of messages."""

    messages: Annotated[List[BaseMessage], add_messages]


# Graph node that makes a stateless request to the Remote Graph Server
def end_node(state: GraphState) -> Dict[str, Any]:
    logger.info(f"Thread end: {state.values()}")
    return {"messages": []}


def llm_node(state: GraphState) -> Dict[str, Any]:
    """
    Creates a plan to solve the user's request

    Args:
        state (State): The current conversation state containing messages and rounds.

    Returns:
        State: The updated state with the assistant's response and incremented rounds.

    Notes:
        - Uses the ChatOpenAI model to generate the assistant's reply.
        - If an error occurs, logs the error and returns a default state.
    """
    prompt = ChatPromptTemplate(
        [
            (
                "system",
                "{system_prompt}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    partial_prompt = prompt.partial(system_prompt=Prompts.SYSTEM)
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"), temperature=1.0)
    generate = partial_prompt | llm

    try:
        llm_response = generate.invoke({"messages": state["messages"]})
        return {"messages": [llm_response]}
    except RuntimeError as e:
        logger.error(f"Error in generation_node: {e}")
        return {"messages": []}


def build_graph() -> Any:
    """
    Constructs the state graph for handling requests.

    Returns:
        StateGraph: A compiled LangGraph state graph.
    """
    builder = StateGraph(GraphState)
    builder.add_node("llm_node", llm_node)
    builder.add_node("end_node", end_node)
    builder.add_edge(START, "llm_node")
    builder.add_edge("llm_node", "end_node")
    builder.add_edge("end_node", END)
    return builder.compile()


def invoke_graph(messages: List[Dict[str, str]], graph: Any = None):
    inputs = {"messages": messages}
    logger.debug({"event": "invoking_graph", "inputs": inputs})
    if not graph:
        graph = build_graph()
    result = graph.invoke(inputs)
    logger.info({"event": "final_result", "result": result})


def main():
    graph = build_graph()
    inputs = {"messages": [HumanMessage(content="Write a story about a cat")]}
    logger.info({"event": "invoking_graph", "inputs": inputs})
    result = graph.invoke(inputs)
    logger.info({"event": "final_result", "result": result})


# Main execution
if __name__ == "__main__":
    invoke_graph([{"role": "user", "content": "write a story about a cat"}])
    # main()
