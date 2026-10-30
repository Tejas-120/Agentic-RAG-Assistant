import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    timeout=30,
    max_retries=0,
)

print("Testing Gemini...")
response = llm.invoke("Reply with only: Gemini works")
print(response.text)