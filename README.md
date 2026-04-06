## Projeto Jurídico — IA Jurídica (MVP)

Este repositório é a base do sistema end-to-end para: **upload de PDF → extração de texto (PyMuPDF/Tesseract) → extração de metadados JSON (IA) → embeddings + busca vetorial (RAG) → predição em dias (XGBoost) → retorno via API**.

### Estrutura do repositório

- `frontend/`: aplicação web (React + TypeScript).
- `backend/`: API e pipeline de ingestão (FastAPI + OCR/extração + integração IA).
- `ml/`: treino e artefatos do modelo (XGBoost) e rotinas de avaliação.

### Variáveis de ambiente

Crie um arquivo `.env` na raiz (use `.env.example` como base). As variáveis abaixo cobrem o MVP local e já deixam o caminho pronto para evoluções.

#### Backend (FastAPI)

- **`BACKEND_HOST`**: host para o servidor (ex: `127.0.0.1`).
- **`BACKEND_PORT`**: porta do servidor (ex: `8000`).
- **`BACKEND_CORS_ORIGINS`**: origens permitidas no CORS (CSV). Ex: `http://localhost:5173,http://127.0.0.1:5173`
- **`BACKEND_LOG_LEVEL`**: nível de log (ex: `INFO`).

#### Extração/OCR

- **`TESSERACT_CMD`**: caminho completo do executável do Tesseract (Windows). Ex: `C:\Program Files\Tesseract-OCR\tesseract.exe`

#### IA (extração de metadados em JSON)

Escolha **um** provedor para começar no MVP.

- **Google Gemini**
  - **`GEMINI_API_KEY`**: chave de API.
  - **`GEMINI_MODEL`**: modelo (ex: `gemini-2.5-flash`).
- **OpenAI**
  - **`OPENAI_API_KEY`**: chave de API.
  - **`OPENAI_MODEL_JSON`**: modelo para extração em JSON (ex: `gpt-4o-mini`).

#### Embeddings (busca vetorial / RAG)

- **`EMBEDDINGS_PROVIDER`**: `openai` | `hf` (HuggingFace) | `local`.
- **`OPENAI_EMBEDDINGS_MODEL`**: modelo de embeddings (ex: `text-embedding-3-small`).
- **`HF_EMBEDDINGS_MODEL`**: modelo HF (ex: `intfloat/multilingual-e5-large`).

#### Vector DB (ChromaDB local)

- **`CHROMA_PERSIST_DIR`**: diretório onde o Chroma persiste os vetores (ex: `./.chroma`).
- **`CHROMA_COLLECTION`**: nome da coleção (ex: `processos`).

#### Banco de dados (PostgreSQL)

- **`DATABASE_URL`**: string de conexão. Ex: `postgresql+psycopg://postgres:postgres@localhost:5432/projeto_juridico`

### Segurança (LGPD) — recomendação mínima

- **Nunca** versionar `.env`, PDFs reais ou dumps de banco.
- Antes de enviar texto para provedores externos, **mascarar CPFs e nomes** (regex) conforme a diretriz do projeto.

