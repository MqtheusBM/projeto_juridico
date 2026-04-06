Modelo de Dados: Estrutura SQL e Vetorial

Este documento descreve a arquitetura da base de dados para o sistema de predição jurídica, dividida entre dados relacionais (PostgreSQL) e dados de alta dimensão (Vector Store).

1. Diagrama Entidade-Relacionamento (ERD)

erDiagram
    USUARIOS ||--o{ PROCESSOS_ENVIADOS : "faz upload"
    PROCESSOS_HISTORICOS ||--|{ SENTENCAS : "possui"
    PROCESSOS_ENVIADOS }|--|| PROCESSOS_HISTORICOS : "similar a"

    USUARIOS {
        uuid id PK
        string nome
        string email
        datetime criado_em
    }

    PROCESSOS_ENVIADOS {
        uuid id PK
        uuid usuario_id FK
        string numero_processo
        string pdf_url
        float valor_causa
        string comarca
        string juiz
        string assunto
        datetime data_envio
    }

    PROCESSOS_HISTORICOS {
        uuid id PK
        string numero_processo
        string assunto_principal
        string tribunal
        string comarca
        string juiz_sentenciante
        float valor_da_causa
        vector embedding_resumo "1536 dim"
    }

    SENTENCAS {
        uuid id PK
        uuid processo_id FK
        datetime data_inicio
        datetime data_sentenca
        int dias_totais
        string resultado "Procedente/Improcedente"
    }


2. Tabelas Principais (PostgreSQL)

A. Tabela processos_historicos

Esta é a base de conhecimento do sistema (os 10.000 processos mencionados).

id: Identificador único.

assunto_principal: Categoria jurídica (ex: Consumidor, Danos Morais).

valor_da_causa: Valor numérico para o modelo de regressão.

comarca / juiz: Variáveis críticas para prever o tempo, pois cada vara tem um ritmo diferente.

B. Tabela sentencas

Contém o "alvo" (target) da nossa predição.

dias_totais: Calculado como data_sentenca - data_inicio. É este valor que o modelo de Machine Learning tentará prever para novos casos.

3. Base de Dados Vetorial (ChromaDB / Pinecone)

Enquanto o SQL guarda "quem" e "quanto", a base vetorial guarda o "quê" (o significado jurídico).

ID do Vetor: O mesmo id da tabela processos_historicos.

Vetor (Embedding): Uma lista de ~1536 números gerada pela IA (ex: OpenAI text-embedding-3-small) que representa o conteúdo da petição.

Metadados: Guardamos apenas o essencial para filtros rápidos (ex: { "comarca": "Boa Vista", "assunto": "Internet" }).

4. Fluxo de Ligação

O sistema extrai o texto do novo PDF.

Gera o Vetor A desse texto.

O Vector DB compara o Vetor A com todos os vetores da tabela processos_historicos.

Retorna os IDs dos 5 processos mais parecidos.

O sistema faz um SELECT no PostgreSQL usando esses IDs para obter os dias_totais reais que esses casos levaram.

A média desses dias é enviada para o modelo de predição final.

5. Recomendações Técnicas

Índices: Criar índices B-Tree nas colunas comarca e assunto no PostgreSQL para consultas rápidas.

Normalização: Criar tabelas auxiliares para juízes e comarcas para evitar erros de digitação (ex: "TJ-RR" vs "TJRR").