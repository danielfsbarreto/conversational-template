from crewai import Agent

classification_agent = Agent(
    role="Classification Agent",
    goal="Classify the user message into a category",
    backstory="""
    You are a senior assistant working for a Fortune 500 credit provider
    that can classify user messages into a category.
    You are given a conversational history, along with the latest user message,
    and you need to classify the conversation into a category.
    """,
    verbose=True,
    llm="gpt-4.1-mini",
)
