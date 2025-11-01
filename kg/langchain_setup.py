# debug_langsmith.py
import os, sys, textwrap
from pathlib import Path

print("Python:", sys.version)
print("CWD   :", Path.cwd())

# --- 1) Load .env from the *script's* folder to avoid CWD issues
try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    print("Installing python-dotenv is required: pip install python-dotenv")
    raise

env_path = find_dotenv(usecwd=True)
print("find_dotenv(usecwd=True) ->", env_path or "NOT FOUND")
load_dotenv(dotenv_path=env_path or ".env", override=True)

# Mask keys for display
def mask(v: str, keep=6):
    if not v: return None
    return v[:keep] + "…" + v[-4:]

print("ENV SNAPSHOT:")
print("  GOOGLE_API_KEY        =", mask(os.getenv("GOOGLE_API_KEY")))
print("  LANGSMITH_ENDPOINT  =", os.getenv("LANGSMITH_ENDPOINT"))
print("  LANGSMITH_API_KEY     =", mask(os.getenv("LANGSMITH_API_KEY")))
print("  LANGSMITH_PROJECT     =", os.getenv("LANGSMITH_PROJECT") or "(unset)")
print("  LANGCHAIN_ENDPOINT    =", os.getenv("LANGCHAIN_ENDPOINT") or "(default)")

# --- 2) Ping LangSmith & ensure project exists
print("\nLangSmith connectivity check:")
try:
    from langsmith import Client, traceable
    client = Client()
    # If no project set, set a default one so you know where to look
    if not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = "Patent Thesis"
        print("  -> LANGCHAIN_PROJECT not set. Using 'Patent Thesis'")
    client.create_project(os.getenv("LANGCHAIN_PROJECT"))
    print("  OK: Authenticated. Current project:", client.project)
except Exception as e:
    print("  !! LangSmith client error:", repr(e))
    print("  HINT: Check LANGCHAIN_API_KEY and network egress (proxy/VPN).")
    sys.exit(1)

# --- 3) Minimal traced function (guaranteed to appear)
@traceable(name="healthcheck_add")
def add(a: int, b: int) -> int:
    return a + b

print("healthcheck_add result:", add(2, 3))

# --- 4) LangChain/Graph versions + v2 tracer check
try:
    import langchain, langgraph, langsmith
    print("\nVersions:")
    print("  langchain:", langchain.__version__)
    print("  langgraph:", langgraph.__version__)
    print("  langsmith:", langsmith.__version__)
    # Optional: check if tracing_v2 appears enabled
    try:
        from langchain_core.tracers.context import tracing_v2_enabled
        print("  tracing_v2_enabled():", tracing_v2_enabled())
    except Exception:
        pass
except Exception as e:
    print("  !! Version import error:", repr(e))

# --- 5) Run your LCEL chain + agent with LangSmith tracing
print("\nRunning LCEL chain + agent:")
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

prompt = ChatPromptTemplate.from_template(
    "Summarize this patent abstract in one sentence:\n\n{abstract}"
)

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

chain = prompt | llm
summary = chain.invoke({"abstract": "A system that adjusts watering based on soil moisture sensors."})
print("Summary:", getattr(summary, "content", summary))

agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    prompt="You are a helpful assistant."
)

config = RunnableConfig(tags=["thesis", "agent", "debug"], metadata={"env": "local"})
agent_out = agent.invoke({"messages": [("human", "What's the weather in Berlin?")]}, config=config)
print("Agent last message:", agent_out["messages"][-1].content if isinstance(agent_out, dict) else agent_out)
print("\nDONE. Check the LangSmith project:", os.getenv("LANGCHAIN_PROJECT"))
