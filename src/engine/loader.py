from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = []

for n in Path('data/raw/').glob('*.pdf'):
    try:
        loader = PyMuPDFLoader(str(n))
        docs.extend(loader.load())
        print(f"Carregado com sucesso o arquivo {n.name}")

    except Exception as e:
        print(f"Erro ao carregar o arquivo {n.name} - {e}")

# print(f"Total de documentos carregados: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

chunks = splitter.split_documents(docs)
