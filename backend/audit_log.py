"""
Logs de auditoria sem expor dados sensíveis (LGPD).

Por que existe um módulo dedicado:
- Centraliza o que pode ou não ir para disco/provedor de logs.
- Evita que `logger.info(...)` acidentais vazem texto de PDF, nomes ou CPF.

Uso:
- Preferir `audit_event(...)` para eventos de negócio (extração, predição, jobs).
- Nunca passar texto integral de petição; usar tamanhos, hashes ou flags.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Optional

# Logger separado: em produção pode ser roteado para arquivo/SIEM sem misturar stack traces.
audit_logger = logging.getLogger("juridico-audit")

# Padrões mínimos para redação defensiva se algum valor textual cair no log por engano.
_CPF_DOTTED = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
_CPF_PLAIN = re.compile(r"\b\d{11}\b")
_CNJ = re.compile(r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\b")


def redact_pii_in_string(value: str, max_preview: int = 24) -> str:
    """
    Remove/mascara padrões óbvios em uma string já destinada a log.

    Parâmetros:
      value: texto potencialmente sensível.
      max_preview: limite de caracteres após redação (evita logs gigantes).

    Retorno:
      String segura para exibir em diagnóstico humano.
    """

    if not value:
        return ""
    s = _CPF_DOTTED.sub("[CPF]", value)
    s = _CPF_PLAIN.sub("[CPF]", s)
    s = _CNJ.sub("[CNJ]", s)
    if len(s) > max_preview:
        return s[:max_preview] + f"...[len={len(s)}]"
    return s


def sha256_prefix(data: bytes, n: int = 16) -> str:
    """
    Hash parcial do conteúdo para correlacionar eventos sem armazenar o arquivo.

    Parâmetros:
      data: bytes (ex.: PDF).
      n: quantidade de hex chars do prefixo do SHA-256.

    Retorno:
      Prefixo hexadecimal.
    """

    h = hashlib.sha256(data).hexdigest()
    return h[: max(8, min(n, len(h)))]


def safe_metadados_summary(metadados: Any) -> dict[str, Any]:
    """
    Resumo não identificante dos metadados para auditoria.

    Por que não logar campos crus:
    - `partes`, `juiz`, `comarca` podem identificar pessoas ou locais pequenos.
    - `numero_processo` é identificador forte; usamos apenas presença/tamanho.

    Retorno:
      Dicionário apenas com contagens/booleanos/tipos.
    """

    def _has(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, str):
            return bool(v.strip())
        return True

    partes = getattr(metadados, "partes", None)
    autor = getattr(partes, "autor", None) if partes is not None else None
    reu = getattr(partes, "reu", None) if partes is not None else None

    np = getattr(metadados, "numero_processo", None)
    comarca = getattr(metadados, "comarca", None)
    assunto = getattr(metadados, "assunto", None)
    juiz = getattr(metadados, "juiz", None)

    return {
        "has_numero_processo": _has(np),
        "numero_processo_len": len(np.strip()) if isinstance(np, str) else 0,
        "has_valor_causa": getattr(metadados, "valor_causa", None) is not None,
        "has_comarca": _has(comarca),
        "comarca_len": len(comarca.strip()) if isinstance(comarca, str) else 0,
        "has_assunto": _has(assunto),
        "assunto_len": len(assunto.strip()) if isinstance(assunto, str) else 0,
        "has_juiz": _has(juiz),
        "juiz_len": len(juiz.strip()) if isinstance(juiz, str) else 0,
        "has_parte_autor": _has(autor),
        "has_parte_reu": _has(reu),
    }


def audit_event(evento: str, **campos: Any) -> None:
    """
    Registra um evento de auditoria com campos já sanitizados.

    Parâmetros:
      evento: nome estável do evento (ex.: "extract_finished").
      **campos: apenas valores seguros (IDs de tarefa, duração, contagens, hashes parciais).

    Por que não sanitizar automaticamente tudo:
      - Risco de dar falso conforto; o chamador deve passar dados mínimos.
    """

    partes_msg = " ".join(f"{k}={v!r}" for k, v in sorted(campos.items()))
    audit_logger.info("evento=%s %s", evento, partes_msg)


def sanitize_exception_for_client(exc: BaseException) -> str:
    """
    Mensagem genérica para resposta HTTP sem encadear stack trace nem dados de entrada.

    Parâmetros:
      exc: exceção capturada.

    Retorno:
      Texto fixo curto; detalhes ficam só em logs do servidor com traceback.
    """

    _ = exc
    return "Falha interna ao processar a tarefa. Consulte os logs do servidor."


def safe_log_exception(logger: logging.Logger, msg: str, exc: BaseException) -> None:
    """
    Loga exceção com traceback no logger da aplicação, sem repetir corpo de requisição.

    Parâmetros:
      logger: logger técnico (não o de auditoria).
      msg: contexto curto.
      exc: exceção.
    """

    logger.exception("%s (causa: %s)", msg, type(exc).__name__)
