#!/usr/bin/env python
from crewai.flow import Flow, listen, persist, start

from conversational_template.agents import (
    classification_agent,
    customer_relationship_agent,
)
from conversational_template.models import (
    ConversationClassification,
    FlowState,
    Message,
)


@persist(verbose=True)
class ConversationalFlow(Flow[FlowState]):
    @start()
    def classify_user_message(self):
        self.state.conversation_classification = classification_agent.kickoff(
            f"""
            Based off of the following conversation history and the latest user message,
            classify the conversation into a category.
            Conversation history: {self.state.history}
            Latest user message: {self.state.user_message}

            The possible categories are:
            - initial_engagement: when the conversation is just starting - so nothing essential has been discussed yet.
            - in_scope: when the discussion is floating around credits, loans, debt, etc.
            - out_of_scope: when the discussion is deviating from the topic.
            """,
            response_format=ConversationClassification,
        ).pydantic

    @listen(classify_user_message)
    def respond_to_user(self):
        self.state.assistant_message = customer_relationship_agent.kickoff(
            f"""
            Based off of the following conversation history and the latest user message,
            respond to the user in a customer relationship.
            Conversation history: {self.state.history}
            Latest user message: {self.state.user_message}

            Be mindful of the conversation classification, trying to steer it towards the appropriate topic.
            However, do not be pushy in case the customer provides a compelling reason to not pay off their debt at this point.
            Conversation classification:
            {self.state.conversation_classification}

            Additionally, make sure to be concise and to the point in your answers, but not too robotic
            or disrespectful.
            """,
            response_format=Message,
        ).pydantic

    @listen(respond_to_user)
    def increment_history(self):
        self.state.history.extend(
            [self.state.user_message, self.state.assistant_message]
        )

    @listen(increment_history)
    def return_response(self):
        print("=============")
        print(self.state.model_dump())
        print("=============")
        return self.state.model_dump()


def kickoff():
    ConversationalFlow().kickoff(
        inputs={
            "user_message": Message(role="user", content="Olá!"),
        }
    )


def plot():
    ConversationalFlow().plot()


if __name__ == "__main__":
    kickoff()
