from src.core.agent import LLM_TRIAGEM
from src.core.prompts import TRIAGEM_PROMPT
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from src.tools.tests import PERGUNTAS_TESTES

# modelo de saída
class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER","PEDIR_INFO","ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA","MÉDIA","ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)


triagem_chain = LLM_TRIAGEM.with_structured_output(TriagemOut)

def triagem(mensagem: str) -> Dict:
    try:
        saida: TriagemOut = triagem_chain.invoke([
            SystemMessage(content=TRIAGEM_PROMPT),
            HumanMessage(content=mensagem)
        ])
        return saida.model_dump()
    except Exception as e:
        return {"erro": str(e)}

# exibindo decisão
for msg_teste in PERGUNTAS_TESTES:
    print(f"Pergunta: {msg_teste}\n -> Resposta: {triagem(msg_teste)}\n")