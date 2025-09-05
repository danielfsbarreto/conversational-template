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
            - football: when the discussion is about football, especially about Grêmio or Internacional.
            - out_of_scope: when the discussion is deviating from the topic.
            """,
            response_format=ConversationClassification,
        ).pydantic

    @listen(classify_user_message)
    def respond_to_user(self):
        self.state.assistant_message = customer_relationship_agent.kickoff(
            f"""
            Based off of the following conversation history and mostly the latest user message,
            respond to the user.
            Conversation history: {self.state.history}
            Latest user message: {self.state.user_message}

            Be mindful of the conversation classification, trying to steer it towards the appropriate topic.
            However, do not be pushy in case the customer provides a compelling reason to not pay off their debt at this point.
            Conversation classification:
            {self.state.conversation_classification}

            Additionally, make sure to be concise and to the point in your answers, but not too robotic
            or disrespectful.

            EASTER EGG
            ==========
            In case the customer discusses football, particularly about Grêmio or Internacional, use the available tool to
            query Google to know the latest match result, include it in the response and go along with the joke.
            Use queries like "resultados do último jogo do [ADD TEAM HERE]" or something similar.
            However, try to politely steer the conversation back to the topic of debt.
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
            "user_message": Message(
                role="user",
                content="Olá! Não estou muito no mood de falar sobre isso pois o Grêmio tá muito ruim das pernas.",
            ),
        }
    )


def plot():
    ConversationalFlow().plot()


if __name__ == "__main__":
    kickoff()
