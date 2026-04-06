import hashlib
import logging
import math
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("juridico-ia-backend.rag_chroma")

_CHROMA_COLLECTION: Optional[Any] = None
_EMBEDDER: Optional[Any] = None


def _env_str(name: str, default: str) -> str:
    """
    Le o valor de variavel de ambiente e faz fallback.

    Parametros:
      name: nome da variavel.
      default: valor caso a variavel nao esteja definida.
    """
    return os.getenv(name, default).strip()


def _truncate(text: str) -> str:
    """
    Trunca o texto para reduzir custo/latencia e evitar inputs gigantes.
    """
    max_chars = int(_env_str("EMBEDDING_MAX_CHARS", "12000"))
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars]


def _mask_sensitive(text: str) -> str:
    """
    Mascaramento minimo antes de embeddings (LGPD).

    Por que: mesmo que nao exista OCR/NER, CPFs sao um dado sensivel frequente.
    """
    import re

    cpf_regexes = [
        r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",
        r"\b\d{11}\b",
    ]
    out = text or ""
    for rgx in cpf_regexes:
        out = re.sub(rgx, "[CPF_MASCA]", out)
    return out


def build_embedding_text_from_metadados(metadados: Any) -> str:
    """
    Constroi um texto semanticamente util (para embeddings) a partir dos metadados.

    Importante: no MVP, o endpoint so recebe metadados (nao o texto completo do PDF).
    Entao usamos as informacoes mais descritivas para aproximar o significado juridico.
    """
    comarca = getattr(metadados, "comarca", None)
    juiz = getattr(metadados, "juiz", None)
    assunto = getattr(metadados, "assunto", None)
    numero_processo = getattr(metadados, "numero_processo", None)
    valor_causa = getattr(metadados, "valor_causa", None)
    partes = getattr(metadados, "partes", None)

    autor = getattr(partes, "autor", None) if partes is not None else None
    reu = getattr(partes, "reu", None) if partes is not None else None

    def _norm(v: Any) -> str:
        return str(v).strip() if v is not None and str(v).strip() else ""

    lines = [
        f"numero_processo: {_norm(numero_processo)}",
        f"comarca: {_norm(comarca)}",
        f"juiz: {_norm(juiz)}",
        f"assunto: {_norm(assunto)}",
        f"valor_causa: {_norm(valor_causa)}",
        f"autor: {_norm(autor)}",
        f"reu: {_norm(reu)}",
    ]

    text = "\\n".join([ln for ln in lines if ln.split(":", 1)[1].strip()])
    return _mask_sensitive(text)


def build_embedding_text_from_sentence(payload: Any) -> str:
    """
    Constroi texto de embeddings a partir de uma sentenca/processo historico (admin).
    """
    numero_processo = getattr(payload, "numero_processo", None)
    assunto_principal = getattr(payload, "assunto_principal", None)
    tribunal = getattr(payload, "tribunal", None)
    comarca = getattr(payload, "comarca", None)
    juiz_sentenciante = getattr(payload, "juiz_sentenciante", None)
    valor_da_causa = getattr(payload, "valor_da_causa", None)

    def _norm(v: Any) -> str:
        return str(v).strip() if v is not None and str(v).strip() else ""

    lines = [
        f"numero_processo: {_norm(numero_processo)}",
        f"assunto_principal: {_norm(assunto_principal)}",
        f"tribunal: {_norm(tribunal)}",
        f"comarca: {_norm(comarca)}",
        f"juiz_sentenciante: {_norm(juiz_sentenciante)}",
        f"valor_da_causa: {_norm(valor_da_causa)}",
    ]

    text = "\\n".join([ln for ln in lines if ln.split(":", 1)[1].strip()])
    return _mask_sensitive(text)


def _local_embedding(text: str, *, dims: int = 384) -> List[float]:
    """
    Embedding "local" deterministico sem dependencias externas.

    Por que: se o usuario escolher `EMBEDDINGS_PROVIDER=local`, o sistema ainda consegue
    gerar vetores para o Chroma, mantendo o pipeline funcional no MVP.
    """
    text = _truncate(text)
    if not text:
        return [0.0] * dims

    # Usa trigramos (caracter) para aproximar semantica de forma leve.
    t = text.lower()
    tokens: List[str] = []
    for i in range(max(0, len(t) - 2)):
        tokens.append(t[i : i + 3])

    vec = [0.0] * dims
    for tok in tokens:
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % dims
        vec[idx] += 1.0

    # Normalizacao L2 para melhorar comportamento com distancia/cosseno.
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _init_embedders_if_needed() -> None:
    """
    Inicializa o modelo de embeddings conforme `EMBEDDINGS_PROVIDER`.
    """
    global _EMBEDDER
    if _EMBEDDER is not None:
        return

    provider = _env_str("EMBEDDINGS_PROVIDER", "openai").lower()
    if provider == "local":
        _EMBEDDER = "local"
        return

    if provider == "openai":
        # Import atrasado: evita falha na importacao da API se o provider nao estiver em uso.
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Dependencia OpenAI ausente/indisponivel. (causa: {e})") from e
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY nao configurado.")
        _EMBEDDER = OpenAI(api_key=api_key)
        return

    if provider == "hf":
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Dependencia HuggingFace ausente/indisponivel. (causa: {e})") from e
        model_name = _env_str("HF_EMBEDDINGS_MODEL", "intfloat/multilingual-e5-large")
        _EMBEDDER = SentenceTransformer(model_name)
        return

    raise RuntimeError(f"EMBEDDINGS_PROVIDER invalido: {provider}. Use openai|hf|local.")


def embed_text(text: str) -> List[float]:
    """
    Gera embedding para um texto unico.

    Parametros:
      text: texto de entrada (idealmente ja sanitizado).
    Retorno:
      lista de floats (vetor).
    """
    _init_embedders_if_needed()
    provider = _env_str("EMBEDDINGS_PROVIDER", "openai").lower()
    text = _truncate(text)
    if not text:
        # Garante que Chroma nao recebe vazio.
        if provider == "local":
            return [0.0] * 384
        if provider == "openai":
            return [0.0] * 1536
        if provider == "hf":
            try:
                # type: ignore[union-attr] - para o provider hf, _EMBEDDER e um SentenceTransformer.
                dims = int(_EMBEDDER.get_sentence_embedding_dimension())  # type: ignore[attr-defined]
            except Exception:
                dims = 1024
            return [0.0] * dims
        return [0.0] * 1536

    if provider == "local":
        return _local_embedding(text)

    if provider == "openai":
        # OpenAI embeddings retornam lista de floats em `data[0].embedding`.
        model = _env_str("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
        # type: ignore[union-attr] - _EMBEDDER e OpenAI quando provider=openai.
        resp = _EMBEDDER.embeddings.create(model=model, input=text)  # type: ignore[attr-defined]
        return list(resp.data[0].embedding)  # type: ignore[no-any-return]

    if provider == "hf":
        model = _EMBEDDER  # type: ignore[assignment]
        # Para SentenceTransformer, encode retorna numpy array; convertemos para list.
        vec = model.encode([text], normalize_embeddings=True)  # type: ignore[attr-defined]
        # `vec[0]` e um array 1D.
        return [float(x) for x in vec[0]]

    raise RuntimeError("Provider nao suportado (verifique EMBEDDINGS_PROVIDER).")


def _init_chroma_if_needed() -> None:
    """
    Inicializa o ChromaDB Persistente (collection) uma unica vez (lazy).
    """
    global _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is not None:
        return

    try:
        import chromadb  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Dependencia chromadb ausente/indisponivel. (causa: {e})") from e

    persist_dir = _env_str("CHROMA_PERSIST_DIR", "./.chroma")
    collection_name = _env_str("CHROMA_COLLECTION", "processos")

    # Mantemos o Chroma em modo persistente para "memoria" local entre execucoes.
    client = chromadb.PersistentClient(path=persist_dir)

    # Use cosseno para embeddings textuais. (Se provider mudar e gerar dimensoes diferentes,
    # o usuario deve limpar a collection, pois Chroma exige consistencia de dimensoes por collection.)
    _CHROMA_COLLECTION = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_process_embedding(
    processo_id: str,
    embedding_text: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Salva (upsert) um embedding no ChromaDB para um processo historico.
    """
    _init_chroma_if_needed()
    metadata = metadata or {}

    embedding_text = _truncate(embedding_text)
    if not embedding_text.strip():
        # Nao faz sentido inserir um embedding vazio.
        return

    emb = embed_text(embedding_text)

    # type: ignore[union-attr] - _CHROMA_COLLECTION e setado no _init_chroma_if_needed.
    _CHROMA_COLLECTION.upsert(
        ids=[processo_id],
        embeddings=[emb],
        metadatas=[metadata],
        documents=[embedding_text],
    )


def query_top_k_process_ids(
    embedding_text: str,
    *,
    top_k: int = 5,
) -> List[str]:
    """
    Consulta a collection do Chroma e retorna os IDs (top_k) mais similares.
    """
    if not (embedding_text or "").strip():
        return []
    _init_chroma_if_needed()

    emb = embed_text(embedding_text)
    # type: ignore[union-attr] - _CHROMA_COLLECTION e setado no _init_chroma_if_needed.
    resp = _CHROMA_COLLECTION.query(
        query_embeddings=[emb],
        n_results=top_k,
        include=["ids", "distances"],
    )

    ids = resp.get("ids", [[]])[0] or []
    # ids pode conter None dependendo do provider; garantimos tipo str.
    out: List[str] = []
    for i in ids:
        if i is None:
            continue
        out.append(str(i))
    return out

