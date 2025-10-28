import anthropic
from langsmith import traceable
from langsmith.wrappers import wrap_anthropic
from dotenv import load_dotenv
load_dotenv()
client = wrap_anthropic(anthropic.Anthropic())

# You can also wrap the async client as well
# async_client = wrap_anthropic(anthropic.AsyncAnthropic())

@traceable(run_type="tool", name="Retrieve Context")
def my_tool(question: str) -> str:
    return "During this morning's meeting, we solved all world conflict."

@traceable(name="Chat Pipeline")
def chat_pipeline(question: str):
    context = my_tool(question)
    messages = [
        { "role": "user", "content": f"Question: {question}\nContext: {context}"}
    ]
    messages = client.messages.create(
      model="claude-sonnet-4-20250514",
      messages=messages,
      max_tokens=1024,
      system="You are a helpful assistant. Please respond to the user's request only based on the given context."
    )
    return messages

chat_pipeline("Can you summarize this morning's meetings?")