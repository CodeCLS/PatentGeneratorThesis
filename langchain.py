# filename: simple_chain_gemini.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# 1️⃣  Create the prompt template
prompt = ChatPromptTemplate.from_template(
    "Summarize this patent abstract in one sentence:\n\n{abstract}"
)

# 2️⃣  Load Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",   # or "gemini-1.5-flash" for faster output
    temperature=0.3
)

# 3️⃣  Build the chain (LCEL syntax)
chain = prompt | llm

# 4️⃣  Run it
result = chain.invoke({
    "abstract": "A system that adjusts watering based on soil moisture sensors."
})

print("💧 Patent Summary:", result.content)
