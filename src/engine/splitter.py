
from langchain_text_splitters import RecursiveCharacterTextSplitter

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)