import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key,
)

texts = [
    "Brain tumor classification using deep learning.",
    "Soft voting combines predictions from multiple models.",
]

vectors = embeddings.embed_documents(texts)

print("Number of text samples:", len(vectors))
print("Dimensions of each vector:", len(vectors[0]))
print("Embedding test successful!")