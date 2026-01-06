from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from src.engine.splitter import SPLITTER
from langchain_community.vectorstores import FAISS
from src.core.agent import LLM_TRIAGEM, LLM_EMBEDDINGS
from src.core.prompts import RAG_PROMPT
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from typing import Literal, List, Dict

docs = []

for n in Path('data/raw/').glob('*.pdf'):
    try:
        loader = PyMuPDFLoader(str(n))
        docs.extend(loader.load())
        print(f"Carregado com sucesso o arquivo {n.name}")

    except Exception as e:
        print(f"Erro ao carregar o arquivo {n.name} - {e}")

# print(f"Total de documentos carregados: {len(docs)}")

chunks = SPLITTER.split_documents(docs)

vectorstore = FAISS.from_documents(chunks, LLM_EMBEDDINGS)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3, "k": 4}
)

document_chain = create_stuff_documents_chain(LLM_TRIAGEM, RAG_PROMPT)

def perguntar_politica_rag(pergunta: str) -> Dict:
    docs_relacionados = retriever.invoke(pergunta)

    if not docs_relacionados:
        return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}
    
    answer = document_chain.invoke({"input": pergunta,
                                    "context": docs_relacionados})
    
    txt = (answer or "").strip()

    if txt.rstrip(".!?") == "Não sei":
        return {"answer": "Não sei.", "citacoes": [], "contexto_encontrado": False}
    
    return {"answer": txt, "citacoes": docs_relacionados, "contexto_encontrado": True}


# perguntas para testes
testes = ["Posso reembolsar a internet?",
          "Quero mais 5 dias de trabalho remoto. Como faço?",
          "Posso reembolsar curso de treinamento da Alura?",
          "Quantas capivaras tem no Rio Pinheiros?",
          "Abre um chamado por favor, é urgente."]

# exibindo decisão
for msg_teste in testes:
    resposta = perguntar_politica_rag(msg_teste)
    print(f"PERGUNTA: {msg_teste}\n")
    print(f"RESPOSTA: {resposta["answer"]}")
    if resposta["contexto_encontrado"]:
        print(f"CITAÇÕES:\n")
        print(resposta["citacoes"])
        print("----------------------------")