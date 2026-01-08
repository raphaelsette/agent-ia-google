![Status](https://img.shields.io/badge/status-ativo-brightgreen) ![Projeto](https://img.shields.io/badge/projeto-portfólio-purple) ![Linguagem](https://img.shields.io/badge/linguagem-Python-blue)

# 🤖 AI: Triagem & RAG System

Este projeto implementa um fluxo inteligente de atendimento utilizando **IA Generativa** para classificar intenções e responder dúvidas baseadas em políticas internas (PDFs).

## 🚀 Funcionalidades

-   **Triagem Inteligente:** Classifica a entrada do usuário em três categorias:
    -   `AUTO_RESOLVER`
    -   `PEDIR_INFO`
    -   `ABRIR_CHAMADO`

-   **RAG (Retrieval-Augmented Generation):** Busca semântica em documentos PDF para garantir respostas precisas e evitar alucinações.
    
-   **Saída Estruturada:** Utiliza Pydantic para garantir que a triagem sempre retorne um JSON válido.

## 🛠️ Tecnologias Utilizadas

-   **Python 3.12.1**
    
-   **LangChain:** Framework para orquestração da IA.
    
-   **Google Gemini (Flash & Embedding):** Modelos de linguagem e embeddings.
    
-   **FAISS:** Banco de dados vetorial local de alta performance.
    
-   **PyMuPDF:** Extração de texto de arquivos PDF.
    
-   **Pydantic:** Validação de dados e estruturação de saída.
    

## 📋 Pré-requisitos

Antes de começar, você precisará de uma chave de API do Google AI Studio. Obtenha em: [aistudio.google.com](https://aistudio.google.com/).

## 🔧 Configuração

1.  **Clone o repositório:**
```
git clone https://github.com/raphaelsette/agent-ia-google.git
cd agent-ia-google
```

<br>

2. **Crie um ambiente virtual e instale as dependências:**
```
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

<br>

3. **Configure as variáveis de ambiente:** Crie um arquivo `.env` na raiz do projeto:
```
GOOGLE_GEMINI_API_KEY=sua_chave_aqui
GOOGLE_GEMINI_MODEL=gemini-2.5-flash-lite
GOOGLE_GEMINI_TEMPERATURE=0

GOOGLE_GEMINI_EMBEDDINGS_API_KEY=sua_chave_aqui
GOOGLE_GEMINI_EMBEDDINGS_MODEL=models/gemini-embedding-001
```

<br>

4. **Adicione seus documentos:** Coloque os arquivos PDF das políticas da empresa na pasta `data/raw/`.

## 📂 Estrutura do Projeto

`src/core/agent.py`: Configuração dos modelos IA.

`src/core/prompts.py`: Engenharia de prompt para triagem e RAG.

`src/engine/splitter.py`: Lógica de fragmentação de documentos (300 chars / 30 overlap).

`src/tools/tests.py`: Perguntas para os testes.

`main.py` e `loader.py`: Ponto de entrada que executa os testes.


## 🧪 Como Executar

Para rodar os testes de triagem e resposta.

    python main.py
    python -m src.engine.loader

