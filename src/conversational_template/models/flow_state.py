from typing import List, Union

from pydantic import BaseModel

from .message import Message


class FlowState(BaseModel):
    user_message: Union[Message, None] = None
    assistant_message: Union[Message, None] = None
    history: List[Message] = []
    conversation_classification: str = "in_scope"
