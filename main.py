import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# load .env
load_dotenv()

# variaveis do .env
CHAVE_GEMINI = os.getenv('GOOGLE_API_KEY')
MODEL_GEMINI = os.getenv('GOOGLE_GEMINI_MODEL')
TEMPERATURE_GEMINI = os.getenv('GOOGLE_GEMINI_TEMPERATURE')

llm = ChatGoogleGenerativeAI(
    model = MODEL_GEMINI,
    temperature = TEMPERATURE_GEMINI,
    api_key = CHAVE_GEMINI
)

resp_test = llm.invoke("Quem é você? Seja criativo.")
print(resp_test.content)