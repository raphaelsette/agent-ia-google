import os
from dotenv import load_dotenv
from src.core.prompts import TRIAGEM_PROMPT
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
from langchain_core.messages import SystemMessage, HumanMessage

# load .env
load_dotenv()

# modelo de saída
class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER","PEDIR_INFO","ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA","MÉDIA","ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)

# iniciando a chamada
llm_triagem = ChatGoogleGenerativeAI(
    model = os.getenv('GOOGLE_GEMINI_MODEL'),
    temperature = float(os.getenv('GOOGLE_GEMINI_TEMPERATURE')),
    api_key = os.getenv('GOOGLE_API_KEY')
)

triagem_chain = llm_triagem.with_structured_output(TriagemOut)

def triagem(mensagem: str) -> Dict:
    try:
        saida: TriagemOut = triagem_chain.invoke([
            SystemMessage(content=TRIAGEM_PROMPT),
            HumanMessage(content=mensagem)
        ])
        return saida.model_dump()
    except Exception as e:
        return {"erro": str(e)}

# perguntas para testes
testes = ["Posso reembolsar a internet?",
          "Quero mais 5 dias de trabalho remoto. Como faço?",
          "Posso reembolsar curso de treinamento da Alura?",
          "Quantas capivaras tem no Rio Pinheiros?",
          "Abre um chamado por favor, é urgente."]

# exibindo decisão
for msg_teste in testes:
    print(f"Pergunta: {msg_teste}\n -> Resposta: {triagem(msg_teste)}\n")