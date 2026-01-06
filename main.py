import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
from langchain_core.messages import SystemMessage, HumanMessage

# load .env
load_dotenv()

# prompt e regras
TRIAGEM_PROMPT = (
    "Você é um triador do service desk para políticas internas da empresa X. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON estruturado.\n\n"
    "Regras de Decisão:\n"
    "- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: 'Posso reembolsar a Internet do meu Home Office?', 'Como funciona a política de alimentação em viagens?').\n"
    "- **PEDIR_INFO**: Mensagens vagas ou que faltem informações para identificar o tema ou contexto (Ex: 'Preciso de ajuda com uma política', 'Tenho uma dúvida geral').\n"
    "- **ABRIR_CHAMADO**: Pedido de exceção, liberação, aprovação ou acesso especial, ou quanto o usuário explicitamente pedir para abrir um chamado (Ex: 'Quero exceção para trabalhar remoto', 'Solicito liberação", "Por favor, abra um chamado para o RH').\n\n"
    "Analise a urgência com base no impacto descrito pelo usuário."
)

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