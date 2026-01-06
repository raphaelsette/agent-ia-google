import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# load .env
load_dotenv()

LLM_TRIAGEM = ChatGoogleGenerativeAI(
    model = os.getenv('GOOGLE_GEMINI_MODEL'),
    temperature = float(os.getenv('GOOGLE_GEMINI_TEMPERATURE')),
    api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
)

LLM_EMBEDDINGS = GoogleGenerativeAIEmbeddings(
    model = os.getenv('GOOGLE_GEMINI_EMBEDDINGS_MODEL'),
    api_key = os.getenv('GOOGLE_GEMINI_EMBEDDINGS_API_KEY')
)