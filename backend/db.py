import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text  # type: ignore[import-not-found]
from sqlalchemy.engine import Engine  # type: ignore[import-not-found]

logger = logging.getLogger("juridico-ia-backend.db")


def _get_db_url() -> str:
    """
    Obtém a URL do PostgreSQL via DATABASE_URL.

    Por que: padroniza a configuração e mantém compatibilidade com `.env.example`.
    """
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        raise RuntimeError("DATABASE_URL não configurado. Defina no `.env` ou ambiente.")
    return db_url


def get_db_engine() -> Engine:
    """
    Cria engine SQLAlchemy para executar DDL e queries.

    Observação: em MVP usamos `text()` (SQL direto) para reduzir complexidade.
    """
    db_url = _get_db_url()
    return create_engine(db_url, pool_pre_ping=True)


def _default_user_uuid() -> uuid.UUID:
    """
    Gera um UUID determinístico para um usuário "anonimo" (sem autenticação).
    """
    env_id = os.getenv("DEFAULT_USER_ID", "").strip()
    if env_id:
        return uuid.UUID(env_id)
    # Determinístico para que persistência funcione sem auth.
    return uuid.uuid5(uuid.NAMESPACE_URL, "juridico-ia-backend:anonymous-user")


def get_default_user_id() -> uuid.UUID:
    """
    Retorna o UUID do usuário padrão (sem autenticação).
    """
    return _default_user_uuid()


def ensure_default_user(engine: Engine) -> None:
    """
    Garante que existe um registro em `usuarios` para persistirmos `processos_enviados`.
    """
    user_id = _default_user_uuid()
    nome = os.getenv("DEFAULT_USER_NAME", "Anônimo").strip()
    email = os.getenv("DEFAULT_USER_EMAIL", "").strip() or None

    sql = text(
        """
        INSERT INTO usuarios (id, nome, email)
        VALUES (:id, :nome, :email)
        ON CONFLICT (id) DO UPDATE SET
            nome = EXCLUDED.nome,
            email = COALESCE(EXCLUDED.email, usuarios.email)
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, {"id": str(user_id), "nome": nome, "email": email})


def _ddl_pgvector() -> str:
    return """
    -- Para embedding_resumo em formato `vector(1536)`.
    -- Se a extensão não existir no Postgres do usuário, o chamador deve cair no fallback.
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS usuarios (
        id uuid PRIMARY KEY,
        nome text NOT NULL,
        email text UNIQUE,
        criado_em timestamptz NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS processos_enviados (
        id uuid PRIMARY KEY,
        usuario_id uuid NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
        numero_processo text NOT NULL,
        pdf_url text,
        valor_causa double precision,
        comarca text,
        juiz text,
        assunto text,
        data_envio timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT uq_processos_enviados_usuario_numero UNIQUE (usuario_id, numero_processo)
    );

    CREATE INDEX IF NOT EXISTS idx_processos_enviados_comarca ON processos_enviados(comarca);
    CREATE INDEX IF NOT EXISTS idx_processos_enviados_assunto ON processos_enviados(assunto);

    CREATE TABLE IF NOT EXISTS processos_historicos (
        id uuid PRIMARY KEY,
        numero_processo text NOT NULL UNIQUE,
        assunto_principal text,
        tribunal text,
        comarca text,
        juiz_sentenciante text,
        valor_da_causa double precision,
        embedding_resumo vector(1536),
        criado_em timestamptz NOT NULL DEFAULT now(),
        updated_em timestamptz NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS idx_processos_historicos_comarca ON processos_historicos(comarca);
    CREATE INDEX IF NOT EXISTS idx_processos_historicos_assunto_principal ON processos_historicos(assunto_principal);

    CREATE TABLE IF NOT EXISTS sentencas (
        id uuid PRIMARY KEY,
        processo_id uuid NOT NULL REFERENCES processos_historicos(id) ON DELETE CASCADE,
        data_inicio timestamptz NOT NULL,
        data_sentenca timestamptz NOT NULL,
        dias_totais integer NOT NULL CHECK (dias_totais >= 0),
        resultado text CHECK (
            resultado IS NULL OR resultado IN ('Procedente', 'Improcedente', 'Procedente/Improcedente')
        ),
        created_em timestamptz NOT NULL DEFAULT now(),
        updated_em timestamptz NOT NULL DEFAULT now()
    );

    CREATE UNIQUE INDEX IF NOT EXISTS uq_sentencas_processo_id ON sentencas(processo_id);
    CREATE INDEX IF NOT EXISTS idx_sentencas_processo_id ON sentencas(processo_id);
    """


def _ddl_jsonb_fallback() -> str:
    return """
    CREATE TABLE IF NOT EXISTS usuarios (
        id uuid PRIMARY KEY,
        nome text NOT NULL,
        email text UNIQUE,
        criado_em timestamptz NOT NULL DEFAULT now()
    );

    CREATE TABLE IF NOT EXISTS processos_enviados (
        id uuid PRIMARY KEY,
        usuario_id uuid NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
        numero_processo text NOT NULL,
        pdf_url text,
        valor_causa double precision,
        comarca text,
        juiz text,
        assunto text,
        data_envio timestamptz NOT NULL DEFAULT now(),
        CONSTRAINT uq_processos_enviados_usuario_numero UNIQUE (usuario_id, numero_processo)
    );

    CREATE INDEX IF NOT EXISTS idx_processos_enviados_comarca ON processos_enviados(comarca);
    CREATE INDEX IF NOT EXISTS idx_processos_enviados_assunto ON processos_enviados(assunto);

    CREATE TABLE IF NOT EXISTS processos_historicos (
        id uuid PRIMARY KEY,
        numero_processo text NOT NULL UNIQUE,
        assunto_principal text,
        tribunal text,
        comarca text,
        juiz_sentenciante text,
        valor_da_causa double precision,
        embedding_resumo jsonb,
        criado_em timestamptz NOT NULL DEFAULT now(),
        updated_em timestamptz NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS idx_processos_historicos_comarca ON processos_historicos(comarca);
    CREATE INDEX IF NOT EXISTS idx_processos_historicos_assunto_principal ON processos_historicos(assunto_principal);

    CREATE TABLE IF NOT EXISTS sentencas (
        id uuid PRIMARY KEY,
        processo_id uuid NOT NULL REFERENCES processos_historicos(id) ON DELETE CASCADE,
        data_inicio timestamptz NOT NULL,
        data_sentenca timestamptz NOT NULL,
        dias_totais integer NOT NULL CHECK (dias_totais >= 0),
        resultado text CHECK (
            resultado IS NULL OR resultado IN ('Procedente', 'Improcedente', 'Procedente/Improcedente')
        ),
        created_em timestamptz NOT NULL DEFAULT now(),
        updated_em timestamptz NOT NULL DEFAULT now()
    );

    CREATE UNIQUE INDEX IF NOT EXISTS uq_sentencas_processo_id ON sentencas(processo_id);
    CREATE INDEX IF NOT EXISTS idx_sentencas_processo_id ON sentencas(processo_id);
    """


def ensure_db_schema(engine: Engine) -> None:
    """
    Cria o schema do PostgreSQL (tabelas + índices) conforme o modelo de dados.

    Para evitar falhas de ambiente:
    - tenta `pgvector` primeiro;
    - se não existir, cria um fallback com `embedding_resumo jsonb` (a persistência continua).
    """
    pgvector_enabled = os.getenv("PGVECTOR_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y"}
    def _exec_sql_script(conn: Any, script: str) -> None:
        """
        Executa um "script" DDL simples separando por `;`.

        Por que: SQLAlchemy/driver tendem a rejeitar múltiplas instruções em um único `execute()`.
        """
        statements = [s.strip() for s in script.split(";") if s.strip()]
        for stmt in statements:
            conn.exec_driver_sql(stmt)

    with engine.begin() as conn:
        try:
            if pgvector_enabled:
                logger.info("Tentando criar schema com pgvector (vector(1536)).")
                _exec_sql_script(conn, _ddl_pgvector())
                return
        except Exception as e:
            logger.warning(
                "Falha ao criar schema com pgvector (causa: %s). Voltando para fallback jsonb.",
                str(e),
            )

        _exec_sql_script(conn, _ddl_jsonb_fallback())


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def upsert_processo_enviado(
    engine: Engine,
    *,
    usuario_id: uuid.UUID,
    numero_processo: str,
    pdf_url: Optional[str],
    valor_causa: Optional[float],
    comarca: Optional[str],
    juiz: Optional[str],
    assunto: Optional[str],
    data_envio: Optional[datetime] = None,
) -> str:
    """
    Insere ou atualiza um registro em `processos_enviados`.
    """
    processo_enviado_id = uuid.uuid4()
    ts = data_envio or _now_utc()

    sql = text(
        """
        INSERT INTO processos_enviados (
            id, usuario_id, numero_processo, pdf_url, valor_causa,
            comarca, juiz, assunto, data_envio
        )
        VALUES (
            :id, :usuario_id, :numero_processo, :pdf_url, :valor_causa,
            :comarca, :juiz, :assunto, :data_envio
        )
        ON CONFLICT (usuario_id, numero_processo) DO UPDATE SET
            pdf_url = COALESCE(EXCLUDED.pdf_url, processos_enviados.pdf_url),
            valor_causa = COALESCE(EXCLUDED.valor_causa, processos_enviados.valor_causa),
            comarca = COALESCE(EXCLUDED.comarca, processos_enviados.comarca),
            juiz = COALESCE(EXCLUDED.juiz, processos_enviados.juiz),
            assunto = COALESCE(EXCLUDED.assunto, processos_enviados.assunto),
            data_envio = EXCLUDED.data_envio
        RETURNING id
        """
    )
    with engine.begin() as conn:
        row = conn.execute(
            sql,
            {
                "id": str(processo_enviado_id),
                "usuario_id": str(usuario_id),
                "numero_processo": numero_processo,
                "pdf_url": pdf_url,
                "valor_causa": valor_causa,
                "comarca": comarca,
                "juiz": juiz,
                "assunto": assunto,
                "data_envio": ts,
            },
        ).first()
        return str(row[0]) if row else str(processo_enviado_id)


def upsert_processo_historico_com_sentenca(
    engine: Engine,
    *,
    numero_processo: str,
    assunto_principal: Optional[str],
    tribunal: Optional[str],
    comarca: Optional[str],
    juiz_sentenciante: Optional[str],
    valor_da_causa: Optional[float],
    data_inicio: datetime,
    data_sentenca: datetime,
    resultado: Optional[str],
    embedding_resumo: Optional[Any] = None,
) -> Dict[str, str]:
    """
    Garante:
    - `processos_historicos` com base no `numero_processo` (upsert)
    - `sentencas` ligada ao processo (upsert)
    """
    processo_id = uuid.uuid4()
    dias_totais = int((data_sentenca - data_inicio).days)
    if dias_totais < 0:
        raise ValueError("dias_totais precisa ser >= 0. Confira data_inicio/data_sentenca.")

    with engine.begin() as conn:
        # Upsert do processo histórico.
        sql_proc = text(
            """
            INSERT INTO processos_historicos (
                id,
                numero_processo,
                assunto_principal,
                tribunal,
                comarca,
                juiz_sentenciante,
                valor_da_causa,
                embedding_resumo
            )
            VALUES (
                :id,
                :numero_processo,
                :assunto_principal,
                :tribunal,
                :comarca,
                :juiz_sentenciante,
                :valor_da_causa,
                :embedding_resumo
            )
            ON CONFLICT (numero_processo) DO UPDATE SET
                assunto_principal = COALESCE(EXCLUDED.assunto_principal, processos_historicos.assunto_principal),
                tribunal = COALESCE(EXCLUDED.tribunal, processos_historicos.tribunal),
                comarca = COALESCE(EXCLUDED.comarca, processos_historicos.comarca),
                juiz_sentenciante = COALESCE(EXCLUDED.juiz_sentenciante, processos_historicos.juiz_sentenciante),
                valor_da_causa = COALESCE(EXCLUDED.valor_da_causa, processos_historicos.valor_da_causa),
                embedding_resumo = COALESCE(EXCLUDED.embedding_resumo, processos_historicos.embedding_resumo),
                updated_em = now()
            RETURNING id
            """
        )
        row_proc = conn.execute(
            sql_proc,
            {
                "id": str(processo_id),
                "numero_processo": numero_processo,
                "assunto_principal": assunto_principal,
                "tribunal": tribunal,
                "comarca": comarca,
                "juiz_sentenciante": juiz_sentenciante,
                "valor_da_causa": valor_da_causa,
                "embedding_resumo": embedding_resumo,
            },
        ).first()
        processo_id = uuid.UUID(str(row_proc[0])) if row_proc else processo_id

        # Upsert da sentença (única por processo no MVP).
        sql_sent = text(
            """
            INSERT INTO sentencas (
                id,
                processo_id,
                data_inicio,
                data_sentenca,
                dias_totais,
                resultado
            )
            VALUES (
                :id,
                :processo_id,
                :data_inicio,
                :data_sentenca,
                :dias_totais,
                :resultado
            )
            ON CONFLICT (processo_id) DO UPDATE SET
                data_inicio = EXCLUDED.data_inicio,
                data_sentenca = EXCLUDED.data_sentenca,
                dias_totais = EXCLUDED.dias_totais,
                resultado = COALESCE(EXCLUDED.resultado, sentencas.resultado),
                updated_em = now()
            RETURNING id
            """
        )
        sentenca_id = uuid.uuid4()
        row_sent = conn.execute(
            sql_sent,
            {
                "id": str(sentenca_id),
                "processo_id": str(processo_id),
                "data_inicio": data_inicio,
                "data_sentenca": data_sentenca,
                "dias_totais": dias_totais,
                "resultado": resultado,
            },
        ).first()
        sentenca_final_id = str(row_sent[0]) if row_sent else str(sentenca_id)
        return {"processo_id": str(processo_id), "sentenca_id": sentenca_final_id}


def get_casos_similares_persistidos(
    engine: Engine,
    *,
    comarca: Optional[str],
    assunto: Optional[str],
    juiz: Optional[str],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Consulta no PostgreSQL (fallback sem vetores) para obter sentenças históricas.

    Retorna uma lista de casos com:
    - `id` (uuid do processo histórico)
    - `dias_totais` (target real)
    """
    # Em MVP: buscamos por comarca e assunto; juiz entra como filtro adicional quando disponível.
    where_clauses: List[str] = []
    params: Dict[str, Any] = {"limit": limit}

    if comarca:
        where_clauses.append("ph.comarca ILIKE :comarca")
        params["comarca"] = comarca
    if assunto:
        where_clauses.append("ph.assunto_principal ILIKE :assunto_principal")
        params["assunto_principal"] = assunto
    if juiz:
        where_clauses.append("ph.juiz_sentenciante ILIKE :juiz_sentenciante")
        params["juiz_sentenciante"] = juiz

    where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"

    # Ordenamos pelos dias_totais mais recentes/maiores para aproximar "casos semelhantes".
    sql = text(
        f"""
        SELECT
            ph.id AS id,
            s.dias_totais AS dias_totais
        FROM processos_historicos ph
        JOIN sentencas s ON s.processo_id = ph.id
        WHERE {where_sql}
        ORDER BY s.data_sentenca DESC
        LIMIT :limit
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(sql, params).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({"id": str(r[0]), "dias_totais": int(r[1])})
        return out


def get_dias_totais_by_process_ids(
    engine: Engine,
    *,
    process_ids: List[str],
) -> List[Dict[str, Any]]:
    """
    Recupera `dias_totais` no PostgreSQL para uma lista de IDs de processos históricos.

    Por que existe:
    - O Chroma retorna apenas os IDs mais similares.
    - O contrato/fluxo do MVP pede que a gente use esses IDs para buscar o target real (`dias_totais`)
      na tabela relacional.

    Parametros:
      engine: engine SQLAlchemy do PostgreSQL.
      process_ids: lista de UUIDs (como string) vindos do Chroma.

    Retorno:
      Lista de dicts no formato: { "id": <uuid>, "dias_totais": <int> } mantendo a ordem
      (e pulando IDs que não existirem no banco).
    """
    if not process_ids:
        return []

    sql = text(
        """
        SELECT
            ph.id AS id,
            s.dias_totais AS dias_totais
        FROM processos_historicos ph
        JOIN sentencas s ON s.processo_id = ph.id
        WHERE ph.id = ANY(:ids)
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(sql, {"ids": process_ids}).fetchall()

    dias_map: Dict[str, int] = {str(r[0]): int(r[1]) for r in rows}

    out: List[Dict[str, Any]] = []
    for pid in process_ids:
        if pid in dias_map:
            out.append({"id": pid, "dias_totais": dias_map[pid]})
    return out

