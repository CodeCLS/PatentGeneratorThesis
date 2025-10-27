# pip install -qU "langchain[anthropic]" to call the model

from langchain.agents import create_agent
from vertexai import init
import os
print("LangSmith tracing on:", os.getenv("LANGCHAIN_TRACING_V2"))
print("LangSmith key loaded:", bool(os.getenv("LANGCHAIN_API_KEY")))
print("LangSmith project:", os.getenv("LANGCHAIN_PROJECT"))
print("Anthropic key loaded:", bool(os.environ.get("ANTHROPIC_API_KEY")))
init(project="gen-lang-client-0744523860", location="us-central1")

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-3-5-sonnet-latest",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)