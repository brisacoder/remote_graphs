# generated by fastapi-codegen:
#   filename:  openapi.json

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import JSONResponse
from models.models import Any, ErrorResponse, RunCreateStateless, Union


router = APIRouter(tags=["Stateless Runs"])
logger = logging.getLogger(__name__)  # This will be "app.api.routes.<name>"


@router.post(
    "/runs",
    response_model=Any,
    responses={
        "404": {"model": ErrorResponse},
        "409": {"model": ErrorResponse},
        "422": {"model": ErrorResponse},
    },
    tags=["Stateless Runs"],
)
def run_stateless_runs_post(body: RunCreateStateless) -> Union[Any, ErrorResponse]:
    """
    Create Background Run
    """
    try:
        # Convert the validated Pydantic model to a dictionary.
        # Using model_dump() is recommended in Pydantic v2 over the deprecated dict() method.
        payload = body.model_dump()
        logging.debug("Decoded payload: %s", payload)

        # Extract assistant_id from the payload
        agent_id = payload.get("agent_id")
        logging.debug(f"Agent id: {agent_id}")

        # Validate that the assistant_id is not empty.
        if not payload.get("agent_id"):
            msg = "agent_id is required and cannot be empty."
            logging.error(msg)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=msg,
            )

        message_id = ""
        # Validate the config section: ensure that config.tags is a non-empty list.
        if (metadata := payload.get("metadata", None)) is not None:
            message_id = metadata.get("id")

        # -----------------------------------------------
        # Extract the human input content from the payload.
        # We expect the content to be located at: payload["input"]["messages"][0]["content"]
        # -----------------------------------------------

        # Retrieve the 'input' field and ensure it is a dictionary.
        input_field = payload.get("input")
        if not isinstance(input_field, dict):
            raise ValueError("The 'input' field should be a dictionary.")

        # Retrieve the 'messages' list from the 'input' dictionary.
        messages = input_field.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("The 'input.messages' field should be a non-empty list.")

        # Access the first message in the list.
        first_message = messages[0]
        if not isinstance(first_message, dict):
            raise ValueError(
                "The first element in 'input.messages' should be a dictionary."
            )

        # Extract the 'content' from the first message.
        human_input_content = first_message.get("content")
        if human_input_content is None:
            raise ValueError(
                "Missing 'content' in the first message of 'input.messages'."
            )

    except HTTPException as http_exc:
        # Log HTTP exceptions and re-raise them so that FastAPI can generate the appropriate response.
        logging.error("HTTP error during run processing: %s", http_exc.detail)
        raise http_exc

    except Exception as exc:
        # Catch unexpected exceptions, log them, and return a 500 Internal Server Error.
        logging.exception("An unexpected error occurred while processing the run.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=exc,
        )

    messages = {
        "messages": [{"role": "assistant", "content": "Received remote request"}]
    }

    # payload to send to autogen server at /runs endpoint
    payload = {
        "agent_id": agent_id,
        "output": messages,
        "model": "gpt-4o",
        "metadata": {"id": message_id},
    }

    logger.info(f"Payload: {payload}")

    # In a real application, additional processing (like starting a background task) would occur here.
    return JSONResponse(content=payload, status_code=status.HTTP_200_OK)


@router.post(
    "/runs/stream",
    response_model=str,
    responses={
        "404": {"model": ErrorResponse},
        "409": {"model": ErrorResponse},
        "422": {"model": ErrorResponse},
    },
    tags=["Stateless Runs"],
)
def stream_run_stateless_runs_stream_post(
    body: RunCreateStateless,
) -> Union[str, ErrorResponse]:
    """
    Create Run, Stream Output
    """
    pass


@router.post(
    "/runs/wait",
    response_model=Any,
    responses={
        "404": {"model": ErrorResponse},
        "409": {"model": ErrorResponse},
        "422": {"model": ErrorResponse},
    },
    tags=["Stateless Runs"],
)
def wait_run_stateless_runs_wait_post(
    body: RunCreateStateless,
) -> Union[Any, ErrorResponse]:
    """
    Create Run, Wait for Output
    """
    pass
