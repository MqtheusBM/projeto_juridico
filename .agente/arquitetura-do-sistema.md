Arquitetura Visual e Fluxo de Dados

Este documento apresenta a visão estrutural do sistema, detalhando a interação entre o utilizador, o processamento de IA e o armazenamento de dados.

1. Diagrama de Arquitetura (High-Level)

graph TD
    subgraph "Camada de Utilizador (Frontend)"
        A[Interface React/TS] -->|Upload PDF| B[API Gateway FastAPI]
    end

    subgraph "Camada de Processamento (Backend)"
        B --> C{Tipo de PDF?}
        C -->|Digital| D[PyMuPDF: Extração de Texto]
        C -->|Imagem| E[Tesseract OCR: Scan]
        D --> F[Limpador de Texto]
        E --> F
    end

    subgraph "Núcleo de Inteligência Artificial"
        F --> G[IA Generativa: Gemini/GPT-4o-mini]
        G --> H[Extração de Metadados JSON: Valor, Juiz, Comarca]
        F --> I[Modelo de Embeddings]
        I --> J[Busca por Similaridade]
    end

    subgraph "Camada de Dados"
        H --> K[(PostgreSQL: Estruturado)]
        J --> L[(Vector DB: ChromaDB/Pinecone)]
        L --> M[Recupera Casos Similares]
    end

    subgraph "Predição Final"
        H --> N[Modelo de Regressão: XGBoost]
        M --> N
        N --> O[Dashboard de Resposta]
        O -->|Visualização| A
    end


2. Detalhes dos Componentes

A. Fluxo de Entrada (Ingestion)

O utilizador interage com uma zona de Drag & Drop no React. O ficheiro é enviado para o FastAPI, que decide instantaneamente se o PDF é legível ou se precisa de passar pelo motor de OCR.

B. O "Pipeline" de Extração

Aqui ocorre a magia da IA Generativa. Enviamos o texto para o Gemini com um comando (prompt) específico:
"Extrai o Valor da Causa e a Comarca deste texto jurídico e responde apenas em JSON."

C. Sistema de RAG (Retrieval-Augmented Generation)

Transformação: O texto da petição inicial é transformado num vetor.

Comparação: O sistema busca processos no banco de dados com vetores "próximos".

Contexto: Permite identificar casos idênticos já julgados na mesma região.

D. O Motor de Cálculo (Machine Learning)

O XGBoost cuida dos números, recebendo os metadados extraídos e a média de tempo dos processos similares para gerar uma estimativa estatística precisa.

3. Topologia de Rede (Ambiente de Produção)

Frontend: Hospedado na Vercel ou AWS Amplify.

Backend: Contentorizado com Docker (AWS ECS ou Google Cloud Run).

Processamento de PDF: Worker assíncrono (Celery/Redis) para grandes volumes.

Base de Dados: Instância gerenciada (Supabase para PostgreSQL + Pinecone para Vetores).

4. Segurança e Privacidade

Anonimização: Remoção de nomes de pessoas físicas e CPFs antes de enviar para APIs externas.

Criptografia: PDFs encriptados at rest e in transit (SSL/TLS).