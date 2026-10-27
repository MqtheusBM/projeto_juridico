import hashlib
import json
import logging
import os
import re
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    # Permite executar a API também quando as variáveis já foram configuradas no ambiente.
    pass

from datajud import DataJudError, search_comparable_cases

# Nota: bibliotecas opcionais (OCR/LLM/embeddings/ML) podem não estar instaladas no MVP local.
# O endpoint deve falhar "gracioso": em vez de quebrar a API, ele retorna `null` (None) quando não
# for possível extrair algo.

from audit_log import (
    audit_event,
    safe_metadados_summary,
    sanitize_exception_for_client,
    sha256_prefix,
)

logger = logging.getLogger("juridico-ia-backend")

# Engine do PostgreSQL (inicializada no startup quando DATABASE_URL estiver configurado).
DB_ENGINE: Optional[Any] = None
DB_IMPORT_ERROR: Optional[str] = None

# Bundle joblib do XGBoost (opcional — carregado no startup se o arquivo existir).
XGBOOST_BUNDLE: Optional[Any] = None
XGBOOST_LOAD_DETAIL: Optional[str] = None

try:
    from db import (  # tipo: ignore
        ensure_db_schema,
        get_db_engine,
        ensure_default_user,
        get_default_user_id,
        upsert_processo_enviado,
        get_casos_similares_persistidos,
        get_dias_totais_by_process_ids,
        upsert_processo_historico_com_sentenca,
    )
except Exception as e:
    # Mantém API funcionando mesmo sem dependências/config de banco (fallback do MVP).
    DB_IMPORT_ERROR = str(e)


# RAG / ChromaDB (opcional no MVP).
RAG_IMPORT_ERROR: Optional[str] = None
RAG_AVAILABLE = False
try:
    from rag_chroma import (  # tipo: ignore
        build_embedding_text_from_metadados,
        build_embedding_text_from_sentence,
        query_top_k_process_ids,
        upsert_process_embedding,
    )

    RAG_AVAILABLE = True
except Exception as e:
    RAG_IMPORT_ERROR = str(e)

try:
    from xgboost_inference import (  # type: ignore
        load_xgboost_bundle,
        predict_dias_with_bundle,
        similaridade_rag_from_casos,
    )
except Exception as e:
    # Causa: ambiente sem dependências de ML ou falha ao importar o módulo local.
    _xgb_inf_import_err = str(e)

    def load_xgboost_bundle(*args: Any, **kwargs: Any) -> tuple[Optional[Any], Optional[str]]:
        return None, _xgb_inf_import_err

    def predict_dias_with_bundle(*args: Any, **kwargs: Any) -> float:
        return 1.0

    def similaridade_rag_from_casos(  # type: ignore
        casos_dias: List[int],
        *,
        global_train_mean_dias: Optional[float],
    ) -> float:
        if casos_dias:
            return float(sum(casos_dias)) / float(len(casos_dias))
        if global_train_mean_dias is not None:
            return float(global_train_mean_dias)
        return 365.0


def _configure_logging() -> None:
    """
    Configura logging usando BACKEND_LOG_LEVEL.

    Por que: o front/end de produção costuma depender de logs consistentes e legíveis.
    """

    level_str = os.getenv("BACKEND_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    # Garante que eventos de auditoria respeitem o mesmo nível configurável.
    logging.getLogger("juridico-audit").setLevel(level)


def _parse_cors_origins(value: str) -> List[str]:
    """
    Converte CSV de origens para lista.

    Parâmetros:
      value: Ex: "http://localhost:5173,http://127.0.0.1:5173"
    """

    parts = [p.strip() for p in value.split(",")]
    return [p for p in parts if p]


# [INÍCIO: CONTRATOS DE RESPOSTA]
class Partes(BaseModel):
    autor: Optional[str] = Field(default=None)
    reu: Optional[str] = Field(default=None)


class MetadadosExtracted(BaseModel):
    numero_processo: Optional[str] = Field(default=None)
    valor_causa: Optional[float] = Field(default=None)
    comarca: Optional[str] = Field(default=None)
    juiz: Optional[str] = Field(default=None)
    assunto: Optional[str] = Field(default=None)
    partes: Partes = Field(default_factory=Partes)


class ExtractResponse(BaseModel):
    metadados: MetadadosExtracted
    # timestamp ajuda debug/auditoria sem vazar dados sensíveis (LGPD).
    gerado_em: datetime = Field(default_factory=datetime.utcnow)


class CasoSimilar(BaseModel):
    id: str
    dias_totais: int


class PredictResponse(BaseModel):
    dias_previstos: float
    casos_similares: List[CasoSimilar]
    gerado_em: datetime = Field(default_factory=datetime.utcnow)


class AsyncJobAccepted(BaseModel):
    """
    Resposta imediata quando o trabalho pesado vai para a fila Celery (V1).
    """

    task_id: str
    status: str = "PENDING"


class JobStatusResponse(BaseModel):
    """
    Estado de uma tarefa assíncrona (sem incluir dados sensíveis em caso de erro).
    """

    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
# [FIM: CONTRATOS DE RESPOSTA]


class SentencaIn(BaseModel):
    """
    Entrada para persistir (processo histórico + sentença).

    Observação: este endpoint é um "admin" do MVP para alimentar treino/RAG.
    """

    numero_processo: str
    data_inicio: datetime
    data_sentenca: datetime

    # Metadados descritivos (opcionais no MVP).
    assunto_principal: Optional[str] = Field(default=None)
    tribunal: Optional[str] = Field(default=None)
    comarca: Optional[str] = Field(default=None)
    juiz_sentenciante: Optional[str] = Field(default=None)
    valor_da_causa: Optional[float] = Field(default=None)

    resultado: Optional[str] = Field(default=None)


class SentencaResponse(BaseModel):
    processo_id: str
    sentenca_id: str
    dias_totais: int
    gerado_em: datetime = Field(default_factory=datetime.utcnow)


class DataJudSearchRequest(BaseModel):
    tribunal: str = Field(default="api_publica_tjrr", min_length=8, max_length=32)
    termos: str = Field(min_length=3, max_length=160)
    limite: int = Field(default=10, ge=1, le=10)


app = FastAPI(
    title="IA Jurídica - API",
    version="0.1.0",
)


_configure_logging()


# [INÍCIO: CORS]
cors_origins = _parse_cors_origins(os.getenv("BACKEND_CORS_ORIGINS", "http://localhost:5173"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)
# [FIM: CORS]


@app.middleware("http")
async def _middleware_auditoria_http(request: Request, call_next: Any) -> Any:
    """
    Registra cada requisição HTTP com duração e correlacionador, sem corpo nem querystring.

    Por que: auditoria operacional (LGPD) — paths e status não expõem conteúdo de petições.
    """

    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        audit_event(
            "http_request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            duration_ms=int((time.perf_counter() - started) * 1000),
            error_type=type(exc).__name__,
        )
        raise

    audit_event(
        "http_request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )
    response.headers["X-Request-ID"] = request_id
    return response


def bootstrap_backend_runtime() -> None:
    """
    Inicializa PostgreSQL e o bundle XGBoost neste processo.

    Por que separado dos eventos FastAPI:
    - O worker Celery também precisa da mesma inicialização e não dispara `startup` do Starlette.
    """

    global DB_ENGINE, XGBOOST_BUNDLE, XGBOOST_LOAD_DETAIL

    if DB_IMPORT_ERROR:
        logger.warning("Camada de persistência indisponível. (causa: %s)", DB_IMPORT_ERROR)
    else:
        try:
            engine = get_db_engine()
            ensure_db_schema(engine)
            ensure_default_user(engine)
            DB_ENGINE = engine
            logger.info("PostgreSQL inicializado e pronto para persistência.")
        except Exception as e:
            logger.warning("Falha ao inicializar PostgreSQL. (causa: %s)", str(e))

    bundle, err = load_xgboost_bundle()
    if bundle is not None:
        XGBOOST_BUNDLE = bundle
        logger.info("Modelo XGBoost disponível para inferência em /api/predict.")
        XGBOOST_LOAD_DETAIL = None
        return

    XGBOOST_LOAD_DETAIL = err
    logger.info(
        "Inferência XGBoost desativada (causa: %s). Usando média RAG/SQL ou heurística.",
        err,
    )


@app.on_event("startup")
def _startup_backend() -> None:
    """Ponto único de bootstrap para a API HTTP."""

    bootstrap_backend_runtime()


def _mask_sensitive(text: str) -> str:
    """
    Mascaramento mínimo (LGPD) antes de enviar texto para serviços externos.

    Por que: mesmo em MVP, a regra pede remover/mascarar CPFs e nomes.
    Neste MVP, implementamos pelo menos CPF com regex (nome pode exigir abordagem
    mais sofisticada/NER, que fica para evolução).
    """

    cpf_regexes = [
        r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b",
        r"\b\d{11}\b",
    ]
    out = text
    for rgx in cpf_regexes:
        out = re.sub(rgx, "[CPF_MASCA]", out)
    return out


def _try_extract_text_pymupdf(pdf_bytes: bytes) -> Optional[str]:
    """
    Tenta extrair texto de PDF digital usando PyMuPDF (fitz).

    Retorna:
      texto (str) se o PDF parecer "digital" (tiver texto selecionável o suficiente);
      caso contrário None (para permitir fallback para OCR).
    """

    try:
        import fitz  # type: ignore
    except Exception:
        return None

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        # Heurística "digital vs scan":
        # - PDFs digitais tendem a ter texto extraível em várias páginas
        # - PDFs escaneados tendem a retornar string vazia/quase vazia
        # Observação: no MVP, preferimos uma heurística simples e tolerante.
        texts: List[str] = []
        pages_with_text = 0
        total_text_chars = 0
        total_pages = getattr(doc, "page_count", None) or 0

        # Regex para detectar "texto real" (letras/números), reduz false positives
        # de strings com apenas pontuação/artefatos.
        real_text_pattern = re.compile(r"[A-Za-zÀ-ÿ0-9]")

        for page in doc:
            page_text = page.get_text("text") or ""
            cleaned = page_text.strip()
            texts.append(cleaned)

            if cleaned and real_text_pattern.search(cleaned):
                pages_with_text += 1
                total_text_chars += len(cleaned)

        full_text = "\n".join(texts).strip()

        # Critérios:
        # 1) Sem "texto real" em nenhuma página -> provavelmente scan.
        # 2) Mesmo com alguma extração, se a fração de páginas com texto é muito baixa
        #    e o volume total é pequeno, preferimos OCR.
        if pages_with_text == 0:
            return None
        if total_text_chars < 50:
            return None
        if total_pages > 0:
            fraction = pages_with_text / float(total_pages)
            if fraction < 0.2:
                return None

        return full_text
    except Exception:
        logger.exception("Falha ao extrair texto via PyMuPDF.")
        return None


def _try_extract_text_tesseract(pdf_bytes: bytes) -> Optional[str]:
    """
    Tenta extrair texto de PDF escaneado usando Tesseract OCR.

    Retorna:
      texto (str) ou None.
    """

    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
        from pdf2image import convert_from_bytes  # type: ignore
    except Exception:
        return None

    try:
        tess_cmd = os.getenv("TESSERACT_CMD")
        if tess_cmd:
            pytesseract.pytesseract.tesseract_cmd = tess_cmd

        images: List[Image.Image] = convert_from_bytes(pdf_bytes)
        texts: List[str] = []
        for img in images:
            texts.append(pytesseract.image_to_string(img))
        text = "\n".join(texts).strip()
        if len(text) < 50:
            return None
        return text
    except Exception:
        logger.exception("Falha ao extrair texto via Tesseract.")
        return None


def _extract_value_causa(text: str) -> Optional[float]:
    """
    Extrai valor da causa com regex baseada em padrões comuns.

    Por que: evita depender 100% de LLM no MVP e garante contrato (null quando ausente).
    """

    # Ex: "Dá-se à causa o valor de R$ 12.345,67"
    patterns = [
        r"valor\s+de\s+R\$\s*([0-9\.\,]+)",
        r"Valor\s+da\s+causa\s*[:\-]?\s*R\$\s*([0-9\.\,]+)",
        r"D[aá]-se\s+a\s+causa\s+o\s+valor\s+de\s+R\$\s*([0-9\.\,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            raw = match.group(1)
            normalized = raw.replace(".", "").replace(",", ".")
            try:
                return float(normalized)
            except ValueError:
                return None
    return None


def _safe_str(s: Optional[str]) -> Optional[str]:
    """
    Normaliza string extraída (trim) e devolve None quando vazia.
    """

    if s is None:
        return None
    s = s.strip()
    return s if s else None


def _extract_metadata_by_heuristics(text: str) -> Dict[str, Any]:
    """
    Extrai metadados com heurísticas simples.

    Observação: no MVP "end-to-end", a extração real será via LLM, mas o endpoint
    precisa já cumprir o contrato de retorno (com nulls quando não achar).
    """

    meta: Dict[str, Any] = {}

    # [INÍCIO: padrões comuns do cabeçalho]
    numero_proc_match = re.search(
        r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\b",
        text,
    )
    meta["numero_processo"] = _safe_str(numero_proc_match.group(0)) if numero_proc_match else None

    meta["valor_causa"] = _extract_value_causa(text)

    comarca_match = re.search(r"Comarca\s*[:\-]?\s*([A-Za-zÀ-ÿ0-9\s\-]+)", text, flags=re.IGNORECASE)
    meta["comarca"] = _safe_str(comarca_match.group(1)) if comarca_match else None

    juiz_match = re.search(r"Juiz\s*[:\-]?\s*([A-Za-zÀ-ÿ\s\-\.]+)", text, flags=re.IGNORECASE)
    meta["juiz"] = _safe_str(juiz_match.group(1)) if juiz_match else None

    assunto_match = re.search(
        r"Assunto\s*[:\-]?\s*([A-Za-zÀ-ÿ0-9\s\-]+)", text, flags=re.IGNORECASE
    )
    meta["assunto"] = _safe_str(assunto_match.group(1)) if assunto_match else None

    autor_match = re.search(
        r"(?:Autor|Requerente)\s*[:\-]?\s*([A-Za-zÀ-ÿ\s\.\-]+)", text, flags=re.IGNORECASE
    )
    reu_match = re.search(
        r"(?:Réu|Requerido|R[eé]u)\s*[:\-]?\s*([A-Za-zÀ-ÿ\s\.\-]+)", text, flags=re.IGNORECASE
    )
    meta["partes"] = {
        "autor": _safe_str(autor_match.group(1)) if autor_match else None,
        "reu": _safe_str(reu_match.group(1)) if reu_match else None,
    }
    # [FIM: padrões comuns do cabeçalho]

    return meta


def _parse_json_lenient(text: str) -> Any:
    """
    Faz parse de JSON mesmo quando a resposta vem "suja".

    Por que: mesmo com `response_mime_type="application/json"`, algumas integrações
    podem devolver texto com prefixos/sufixos. O objetivo aqui é tornar o endpoint
    mais tolerante, sem aceitar dados "inventados".
    """
    raw = (text or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        # [INÍCIO: tentativa de extração do bloco {...}]
        # Captura do primeiro "{" até o último "}".
        m = re.search(r"\{[\s\S]*\}$", raw)
        if not m:
            raise
        candidate = m.group(0)
        # [FIM: tentativa de extração do bloco {...}]
        return json.loads(candidate)


def _extract_metadados_by_llm(texto_pdf: str) -> Optional[MetadadosExtracted]:
    """
    Extrai metadados usando um provedor de LLM (Gemini) com prompt forçado em JSON.

    Validação de esquema:
    - o resultado final é validado com Pydantic (`MetadadosExtracted`).
    - campos ausentes devem virar `null` (no JSON) / `None` (no Python).

    Retorna:
      - MetadadosExtracted se houver JSON válido e coerente.
      - None se não for possível (para permitir fallback heurístico).
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    # [INÍCIO: inicialização do provedor (falha "graciosa")]
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        logger.info("Biblioteca `google-generativeai` não instalada; usando heurísticas.")
        return None

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name,
            system_instruction=(
                "És um assistente especializado em Direito Brasileiro e Processo Civil. "
                "A tua única tarefa é ler o texto de uma Petição Inicial e extrair metadados específicos. "
                "Deves ignorar argumentos retóricos e focar em factos. "
                "Se um dado não for encontrado, retorna 'null'. "
                "Responde estritamente em formato JSON."
            ),
        )
    except Exception:
        logger.exception("Falha ao inicializar Gemini; usando heurísticas.")
        return None
    # [FIM: inicialização do provedor (falha "graciosa")]

    # [INÍCIO: template do prompt de extração (conforme estrategia-de-prompt.md)]
    schema_example = {
        "numero_processo": "string",
        "valor_causa": 0.0,
        "comarca": "string",
        "juiz": "string",
        "assunto": "string",
        "partes": {"autor": "string", "reu": "string"},
    }

    user_prompt = (
        "Analise o seguinte texto de petição inicial e extraia as informações solicitadas:\n"
        "1. Número do Processo (se houver).\n"
        "2. Valor da Causa (apenas o número).\n"
        "3. Comarca/Tribunal.\n"
        "4. Nome do Juiz (se mencionado no cabeçalho).\n"
        "5. Assunto Principal (ex: Dano Moral, Cobrança Indevida, Voo Cancelado).\n"
        "6. Nome das Partes (Autor e Réu).\n\n"
        "TEXTO DA PETIÇÃO:\n"
        f"{texto_pdf}\n\n"
        "RESPONDA APENAS NO FORMATO JSON ABAIXO:\n"
        f"{json.dumps(schema_example, ensure_ascii=False)}\n\n"
        "Técnicas de precisão:\n"
        "- Valor da causa: foque na seção 'Dá-se à causa o valor de...'.\n"
        "- Assunto: se o texto fala sobre bagagem, use 'Transporte Aéreo'.\n"
        "- Se um dado não for encontrado, retorna null.\n"
        "- Não inclua texto fora do JSON.\n"
    )
    # [FIM: template do prompt de extração]

    # [INÍCIO: chamada e validação do JSON]
    try:
        response = model.generate_content(
            user_prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        response_text = getattr(response, "text", None) or str(response)
        payload = _parse_json_lenient(response_text)
        validated = MetadadosExtracted.model_validate(payload)
        return validated
    except Exception as e:
        # Causa mais comum: JSON inválido ou incompatibilidade de tipo.
        logger.warning("Falha ao extrair metadados via LLM (causa: %s).", str(e))
        return None
    # [FIM: chamada e validação do JSON]


def _extract_metadados(texto_pdf: str) -> MetadadosExtracted:
    """
    Extrai metadados a partir de texto de PDF.

    Estratégia (MVP):
    1) Tenta aplicar extração via LLM com prompt forçado em JSON.
    2) Se falhar (biblioteca/API/validação), faz fallback para heurísticas locais.
    """

    texto_sanitizado = _mask_sensitive(texto_pdf)

    # [INÍCIO: extração via LLM]
    metadados_llm = _extract_metadados_by_llm(texto_sanitizado)
    if metadados_llm is not None:
        return metadados_llm
    # [FIM: extração via LLM]

    # [INÍCIO: fallback heurístico]
    heur = _extract_metadata_by_heuristics(texto_sanitizado)

    # Pydantic vai garantir o contrato: campos ausentes viram `null` (None).
    return MetadadosExtracted.model_validate(heur)
    # [FIM: fallback heurístico]


def _predict_days_basic(metadados: MetadadosExtracted) -> float:
    """
    Predição básica (fallback) para garantir que o endpoint retorna dados coerentes.

    Por que: no começo do MVP pode não existir base/ML treinado.
    Assim, devolvemos uma previsão determinística e consistente.
    """

    base = 30.0
    if metadados.valor_causa is not None:
        # Usa log para reduzir impacto de valores muito grandes.
        # (Mantém o resultado em uma faixa razoável para o fallback do MVP.)
        valor = max(0.0, float(metadados.valor_causa))
        base += 10.0 * (1.0 + (valor + 1.0) ** 0.1)
        base += 5.0 * (valor ** 0.05)

    # Ajuste por "assunto"/comarca (semântica real virá no ML/RAG).
    seed_str = f"{metadados.comarca}|{metadados.assunto}|{metadados.numero_processo}"
    seed = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:8], 16)
    jitter = (seed % 2000) / 100.0  # 0..20
    return float(max(1.0, base + (jitter - 10.0)))


def _make_similar_cases(metadados: MetadadosExtracted, dias_previstos: float) -> List[CasoSimilar]:
    """
    Gera casos similares consistentes no fallback.

    Por que: o contrato pede lista com IDs e valores mesmo sem base vetorial/PostgreSQL.
    """

    case_id = "mock_" + hashlib.sha256(
        f"{metadados.comarca}|{metadados.assunto}|{metadados.juiz}".encode("utf-8")
    ).hexdigest()[:12]
    dias_int = int(round(max(0.0, min(dias_previstos, 3650.0))))
    return [CasoSimilar(id=case_id, dias_totais=dias_int)]


def run_extract_from_pdf_bytes(pdf_bytes: bytes) -> ExtractResponse:
    """
    Núcleo síncrono da extração: PDF -> texto -> metadados -> persistência opcional.

    Parâmetros:
      pdf_bytes: conteúdo bruto do PDF.

    Retorno:
      `ExtractResponse` validado.

    Por que função isolada:
    - Reutilizada pelo endpoint HTTP e pelo worker Celery sem duplicar lógica.
    """

    texto_pdf: Optional[str] = None
    texto_pdf = _try_extract_text_pymupdf(pdf_bytes)
    if texto_pdf is None:
        texto_pdf = _try_extract_text_tesseract(pdf_bytes)

    if texto_pdf is None:
        return ExtractResponse(metadados=MetadadosExtracted())

    metadados = _extract_metadados(texto_pdf)

    try:
        if DB_ENGINE is not None and metadados.numero_processo:
            upsert_processo_enviado(
                DB_ENGINE,
                usuario_id=get_default_user_id(),
                numero_processo=metadados.numero_processo,
                pdf_url=None,
                valor_causa=metadados.valor_causa,
                comarca=metadados.comarca,
                juiz=metadados.juiz,
                assunto=metadados.assunto,
            )
    except Exception as e:
        logger.warning(
            "Falha ao persistir processo enviado no PostgreSQL. (causa: %s)",
            str(e),
        )

    return ExtractResponse(metadados=metadados)


def run_predict_core(
    metadados_in: MetadadosExtracted,
    texto_para_embedding: Optional[str],
) -> PredictResponse:
    """
    Núcleo síncrono da predição: RAG/SQL + XGBoost (ou fallbacks).

    Parâmetros:
      metadados_in: metadados já normalizados.
      texto_para_embedding: texto completo mascarado para embedding (multipart), ou None (JSON).

    Retorno:
      `PredictResponse` com `dias_previstos` e `casos_similares`.
    """

    casos_similares: List[CasoSimilar] = []

    try:
        if DB_ENGINE is not None and metadados_in.numero_processo:
            upsert_processo_enviado(
                DB_ENGINE,
                usuario_id=get_default_user_id(),
                numero_processo=metadados_in.numero_processo,
                pdf_url=None,
                valor_causa=metadados_in.valor_causa,
                comarca=metadados_in.comarca,
                juiz=metadados_in.juiz,
                assunto=metadados_in.assunto,
            )

        if DB_ENGINE is not None:
            rag_ok = False
            if RAG_AVAILABLE and metadados_in is not None:
                try:
                    if texto_para_embedding is not None:
                        embedding_text = _mask_sensitive(texto_para_embedding)
                    else:
                        embedding_text = build_embedding_text_from_metadados(metadados_in)

                    process_ids = query_top_k_process_ids(embedding_text, top_k=5)

                    if process_ids:
                        db_days = get_dias_totais_by_process_ids(
                            DB_ENGINE, process_ids=process_ids
                        )
                        if db_days:
                            casos_similares = [
                                CasoSimilar(id=c["id"], dias_totais=int(c["dias_totais"]))
                                for c in db_days
                            ]
                            rag_ok = True
                except Exception as e:
                    logger.warning("Falha ao usar ChromaDB/RAG. (causa: %s)", str(e))

            if not rag_ok:
                db_cases = get_casos_similares_persistidos(
                    DB_ENGINE,
                    comarca=metadados_in.comarca,
                    assunto=metadados_in.assunto,
                    juiz=metadados_in.juiz,
                    limit=5,
                )
                if db_cases:
                    casos_similares = [
                        CasoSimilar(id=c["id"], dias_totais=int(c["dias_totais"])) for c in db_cases
                    ]
    except Exception as e:
        logger.warning("Falha ao usar PostgreSQL para RAG/Persistência. (causa: %s)", str(e))

    dias_nos_similares = [c.dias_totais for c in casos_similares]
    train_cfg = (
        getattr(XGBOOST_BUNDLE, "training_config", None) or {} if XGBOOST_BUNDLE is not None else {}
    )
    global_mean_opt = train_cfg.get("global_train_mean_dias")
    global_mean_f = float(global_mean_opt) if global_mean_opt is not None else None

    similaridade_rag = similaridade_rag_from_casos(
        dias_nos_similares,
        global_train_mean_dias=global_mean_f,
    )

    dias_previstos: float
    if XGBOOST_BUNDLE is not None:
        try:
            dias_previstos = predict_dias_with_bundle(
                XGBOOST_BUNDLE,
                comarca=metadados_in.comarca,
                juiz=metadados_in.juiz,
                assunto=metadados_in.assunto,
                valor_causa=metadados_in.valor_causa,
                similaridade_rag=similaridade_rag,
            )
        except Exception as e:
            logger.warning("Falha na inferência XGBoost (causa: %s). Usando fallback.", str(e))
            if casos_similares:
                dias_previstos = float(sum(dias_nos_similares)) / float(len(dias_nos_similares))
            else:
                dias_previstos = _predict_days_basic(metadados_in)
                casos_similares = _make_similar_cases(metadados_in, dias_previstos)
    elif casos_similares:
        dias_previstos = float(sum(dias_nos_similares)) / float(len(dias_nos_similares))
    else:
        dias_previstos = _predict_days_basic(metadados_in)
        casos_similares = _make_similar_cases(metadados_in, dias_previstos)

    return PredictResponse(dias_previstos=dias_previstos, casos_similares=casos_similares)


def run_predict_from_pdf_bytes(pdf_bytes: bytes) -> PredictResponse:
    """
    Predição a partir do PDF: extrai texto/metadados e delega para `run_predict_core`.

    Parâmetros:
      pdf_bytes: conteúdo bruto do PDF.
    """

    texto_pdf = _try_extract_text_pymupdf(pdf_bytes) or _try_extract_text_tesseract(pdf_bytes)
    if texto_pdf is None:
        metadados_in = MetadadosExtracted()
        texto_para_embedding = None
    else:
        texto_para_embedding = texto_pdf
        metadados_in = _extract_metadados(texto_pdf)
    return run_predict_core(metadados_in, texto_para_embedding)


def _gravar_pdf_temporario(pdf_bytes: bytes) -> str:
    """
    Persiste bytes do PDF em arquivo temporário para o worker Celery consumir e apagar.

    Parâmetros:
      pdf_bytes: conteúdo do PDF.

    Retorno:
      Caminho absoluto do arquivo criado.

    Por que arquivo em disco:
    - Evita mensagens gigantes no broker Redis e mantém o contrato simples entre API e worker.
    """

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(pdf_bytes)
        return tmp.name
    finally:
        tmp.close()


@app.post("/api/extract", response_model=ExtractResponse)
async def extract_pdf(file: UploadFile = File(...)) -> ExtractResponse:
    """
    Extrai metadados do PDF.

    Entrada:
      file: arquivo PDF enviado pelo frontend.

    Saída:
      JSON com `metadados` validados. Campos não encontrados viram `null`.
    """

    if file.content_type and "pdf" not in file.content_type.lower():
        raise HTTPException(status_code=400, detail="Arquivo deve ser um PDF.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="PDF vazio.")

    audit_event(
        "extract_http_started",
        pdf_sha256_prefix=sha256_prefix(pdf_bytes),
        pdf_size=len(pdf_bytes),
    )
    result = run_extract_from_pdf_bytes(pdf_bytes)
    audit_event("extract_http_finished", **safe_metadados_summary(result.metadados))
    return result


@app.post("/api/datajud/comparaveis")
def datajud_comparaveis(payload: DataJudSearchRequest) -> Dict[str, Any]:
    """Busca amostras públicas por termos e resume uma duração exploratória por etapa."""
    try:
        return search_comparable_cases(
            tribunal=payload.tribunal,
            query=payload.termos,
            limit=payload.limite,
        )
    except DataJudError as exc:
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        raise HTTPException(status_code=exc.status_code, detail=str(exc), headers=headers) from exc


@app.post("/api/predict", response_model=PredictResponse)
async def predict_days(request: Request) -> PredictResponse:
    """
    Prediz dias previstos a partir de metadados (ou PDF).

    Compatibilidade (MVP):
    - Se a request for JSON: lê `metadados`.
    - Se for multipart/form-data: tenta ler `file` e roda extração via `/api/extract` internamente.
    """

    content_type = request.headers.get("content-type", "")
    metadados_in: Optional[MetadadosExtracted] = None
    texto_para_embedding: Optional[str] = None

    if "multipart/form-data" in content_type:
        form = await request.form()
        upload_file = form.get("file")
        # Starlette devolve um `UploadFile` compatível, mas o tipo exato pode variar.
        # Assim, validamos pelo comportamento mínimo esperado: método `.read()`.
        if not upload_file or not hasattr(upload_file, "read"):
            raise HTTPException(status_code=400, detail="multipart precisa enviar `file` (PDF).")

        # Reaproveita a lógica do extract: mantém consistência e reduz duplicação.
        pdf_bytes = await upload_file.read()
        texto_pdf = _try_extract_text_pymupdf(pdf_bytes) or _try_extract_text_tesseract(pdf_bytes)
        if texto_pdf is None:
            metadados_in = MetadadosExtracted()
        else:
            # Usamos o texto extraido para embeddings quando possivel.
            # Por que: embeddings no texto completo tendem a capturar significado juridico melhor
            # do que embeddings apenas a partir de metadados.
            texto_para_embedding = texto_pdf
            metadados_in = _extract_metadados(texto_pdf)
    else:
        payload = await request.json()
        raw_meta = payload.get("metadados")
        if raw_meta is None:
            raise HTTPException(status_code=400, detail="JSON precisa enviar `metadados`.")
        try:
            metadados_in = MetadadosExtracted.model_validate(raw_meta)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"metadados inválidos: {str(e)}")

    if metadados_in is None:
        raise HTTPException(status_code=400, detail="metadados ausentes.")

    audit_event("predict_http_started", **safe_metadados_summary(metadados_in))
    result = run_predict_core(metadados_in, texto_para_embedding)
    audit_event(
        "predict_http_finished",
        dias_previstos=result.dias_previstos,
        casos_similares_count=len(result.casos_similares),
    )
    return result


@app.post("/api/extract/async", response_model=AsyncJobAccepted, status_code=202)
async def extract_pdf_async(file: UploadFile = File(...)) -> AsyncJobAccepted:
    """
    Enfileira extração pesada (OCR/LLM) no Celery.

    Saída:
      `task_id` para acompanhar em `GET /api/jobs/{task_id}`.

    Causa comum de 503:
      - Redis indisponível ou worker Celery não iniciado.
    """

    if file.content_type and "pdf" not in file.content_type.lower():
        raise HTTPException(status_code=400, detail="Arquivo deve ser um PDF.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="PDF vazio.")

    path = _gravar_pdf_temporario(pdf_bytes)
    try:
        from tasks import extract_pdf_from_path_task

        async_result = extract_pdf_from_path_task.delay(path)
    except Exception as e:
        try:
            os.unlink(path)
        except OSError:
            pass
        logger.warning(
            "Falha ao enfileirar extração assíncrona. (causa: %s)",
            type(e).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="Fila assíncrona indisponível. Verifique Redis, dependências Celery e o worker.",
        )

    audit_event(
        "extract_async_accepted",
        task_id=async_result.id,
        pdf_sha256_prefix=sha256_prefix(pdf_bytes),
        pdf_size=len(pdf_bytes),
    )
    return AsyncJobAccepted(task_id=async_result.id)


@app.post("/api/predict/async", response_model=AsyncJobAccepted, status_code=202)
async def predict_async(request: Request) -> AsyncJobAccepted:
    """
    Enfileira predição assíncrona (JSON com `metadados` ou multipart com PDF).

    Compatível com o mesmo formato de `POST /api/predict`.
    """

    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        upload_file = form.get("file")
        if not upload_file or not hasattr(upload_file, "read"):
            raise HTTPException(status_code=400, detail="multipart precisa enviar `file` (PDF).")

        pdf_bytes = await upload_file.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="PDF vazio.")

        path = _gravar_pdf_temporario(pdf_bytes)
        try:
            from tasks import predict_from_pdf_path_task

            async_result = predict_from_pdf_path_task.delay(path)
        except Exception as e:
            try:
                os.unlink(path)
            except OSError:
                pass
            logger.warning(
                "Falha ao enfileirar predição assíncrona (PDF). (causa: %s)",
                type(e).__name__,
            )
            raise HTTPException(
                status_code=503,
                detail="Fila assíncrona indisponível. Verifique Redis, dependências Celery e o worker.",
            )

        audit_event(
            "predict_async_pdf_accepted",
            task_id=async_result.id,
            pdf_sha256_prefix=sha256_prefix(pdf_bytes),
            pdf_size=len(pdf_bytes),
        )
        return AsyncJobAccepted(task_id=async_result.id)

    payload = await request.json()
    raw_meta = payload.get("metadados")
    if raw_meta is None:
        raise HTTPException(status_code=400, detail="JSON precisa enviar `metadados`.")
    try:
        metadados_in = MetadadosExtracted.model_validate(raw_meta)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"metadados inválidos: {str(e)}")

    try:
        from tasks import predict_from_metadados_task

        async_result = predict_from_metadados_task.delay(
            metadados_in.model_dump(mode="json"),
        )
    except Exception as e:
        logger.warning(
            "Falha ao enfileirar predição assíncrona (JSON). (causa: %s)",
            type(e).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="Fila assíncrona indisponível. Verifique Redis, dependências Celery e o worker.",
        )

    audit_event(
        "predict_async_json_accepted",
        task_id=async_result.id,
        **safe_metadados_summary(metadados_in),
    )
    return AsyncJobAccepted(task_id=async_result.id)


@app.get("/api/jobs/{task_id}", response_model=JobStatusResponse)
def get_job_status(task_id: str) -> JobStatusResponse:
    """
    Consulta o estado de uma tarefa Celery sem expor stack trace ao cliente.
    """

    try:
        from celery.result import AsyncResult

        from celery_app import celery_app
    except Exception as e:
        logger.warning("Celery indisponível para consulta de jobs. (causa: %s)", type(e).__name__)
        raise HTTPException(status_code=503, detail="Celery não configurado ou não instalado.")

    resultado = AsyncResult(task_id, app=celery_app)
    estado = resultado.state

    if estado in ("PENDING", "RECEIVED", "RETRY"):
        return JobStatusResponse(task_id=task_id, status="PENDING")
    if estado == "STARTED":
        return JobStatusResponse(task_id=task_id, status="STARTED")
    if estado == "SUCCESS":
        return JobStatusResponse(task_id=task_id, status="SUCCESS", result=resultado.result)
    if estado == "FAILURE":
        info = resultado.info
        exc_tipo = type(info).__name__ if info is not None else "Failure"
        audit_event("job_status_failure", task_id=task_id, exc_type=exc_tipo)
        return JobStatusResponse(
            task_id=task_id,
            status="FAILURE",
            error=sanitize_exception_for_client(
                info if isinstance(info, BaseException) else Exception()
            ),
        )

    return JobStatusResponse(task_id=task_id, status=str(estado))


@app.post("/api/sentences", response_model=SentencaResponse)
async def upsert_sentence(payload: SentencaIn) -> SentencaResponse:
    """
    Endpoint para persistir processos históricos e sentenças no PostgreSQL.

    Isso alimenta o futuro RAG/treino sem exigir que a aplicação já tenha embeddings/VectorDB.
    """
    if DB_ENGINE is None:
        raise HTTPException(status_code=503, detail="PostgreSQL não inicializado (verifique DATABASE_URL).")

    try:
        dias_totais = int((payload.data_sentenca - payload.data_inicio).days)
        if dias_totais < 0:
            raise ValueError("data_sentenca precisa ser >= data_inicio.")

        result = upsert_processo_historico_com_sentenca(
            DB_ENGINE,
            numero_processo=payload.numero_processo,
            assunto_principal=payload.assunto_principal,
            tribunal=payload.tribunal,
            comarca=payload.comarca,
            juiz_sentenciante=payload.juiz_sentenciante,
            valor_da_causa=payload.valor_da_causa,
            data_inicio=payload.data_inicio,
            data_sentenca=payload.data_sentenca,
            resultado=payload.resultado,
            embedding_resumo=None,
        )

        # [INÍCIO: persistência vetorial (ChromaDB)]
        if RAG_AVAILABLE:
            try:
                embedding_text = build_embedding_text_from_sentence(payload)
                if embedding_text.strip():
                    upsert_process_embedding(
                        result["processo_id"],
                        embedding_text,
                        metadata={
                            "comarca": payload.comarca,
                            "assunto": payload.assunto_principal,
                            "juiz": payload.juiz_sentenciante,
                            "tribunal": payload.tribunal,
                        },
                    )
            except Exception as e:
                logger.warning(
                    "Falha ao salvar embeddings no ChromaDB. (causa: %s)",
                    str(e),
                )
        # [FIM: persistência vetorial (ChromaDB)]

        audit_event(
            "sentence_upsert_http_ok",
            dias_totais=dias_totais,
            processo_id_len=len(str(result.get("processo_id", ""))),
        )
        return SentencaResponse(
            processo_id=result["processo_id"],
            sentenca_id=result["sentenca_id"],
            dias_totais=dias_totais,
        )
    except HTTPException:
        raise
    except Exception as e:
        # Primeiro explicamos a causa antes do motivo da falha.
        raise HTTPException(status_code=400, detail=f"Falha ao persistir sentença (causa: {str(e)}).")

