from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"] = Field(
        ..., description="The role of the message sender, either 'user' or 'assistant'."
    )
    content: str = Field(..., description="The textual content of the message.")
