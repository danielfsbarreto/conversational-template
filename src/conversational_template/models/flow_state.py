from typing import List, Union

from pydantic import BaseModel

from .conversation_classification import ConversationClassification
from .message import Message


class FlowState(BaseModel):
    user_message: Union[Message, None] = None
    assistant_message: Union[Message, None] = None
    history: List[Message] = []
    conversation_classification: Union[ConversationClassification, None] = None
