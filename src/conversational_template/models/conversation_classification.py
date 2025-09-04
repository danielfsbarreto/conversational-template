from typing import Literal

from pydantic import BaseModel, Field


class ConversationClassification(BaseModel):
    category: Literal["initial_engagement", "in_scope", "out_of_scope"] = Field(
        ...,
        description="The classification category of the conversation: 'initial_engagement' for initial engagement, 'in_scope' for relevant topics, 'out_of_scope' otherwise.",
    )
    score: float = Field(
        ...,
        description="A confidence score for the classification, always between 0 and 1.",
    )
    reasoning: str = Field(
        ...,
        description="Explanation or reasoning for why the conversation was classified as such.",
    )
