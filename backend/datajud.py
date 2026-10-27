"""Cliente mínimo da API Pública DataJud do CNJ.

Mantém o host fixo e aceita apenas aliases no formato documentado pelo CNJ, para
que parâmetros do usuário não possam redirecionar a requisição para outro host.
"""

import json
import os
import re
import sqlite3
import statistics
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DATAJUD_BASE_URL = "https://api-publica.datajud.cnj.jus.br"
ALIAS_RE = re.compile(r"^api_publica_[a-z0-9]{2,12}$")
CACHE_TTL_SECONDS = 600
PERSISTENT_CACHE_DEFAULT_TTL_SECONDS = 21600
_CACHE_LOCK = threading.Lock()
_SEARCH_CACHE: Dict[tuple, tuple] = {}
_RATE_LIMIT_UNTIL = 0.0


class DataJudError(RuntimeError):
    """Erro de configuração ou indisponibilidade do DataJud."""

    def __init__(self, message: str, *, status_code: int = 503, retry_after: int = 0):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = max(0, retry_after)


def _extract_source(hit: Dict[str, Any]) -> Dict[str, Any]:
    source = hit.get("_source")
    return source if isinstance(source, dict) else {}


def _names(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    return [str(item["nome"]) for item in items if isinstance(item, dict) and item.get("nome")]


def _movements(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append({
            "codigo": item.get("codigo"),
            "nome": item.get("nome"),
            "data_hora": item.get("dataHora"),
        })
    return out


def _normalize_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    source = _extract_source(hit)
    classe = source.get("classe") if isinstance(source.get("classe"), dict) else {}
    tribunal = source.get("tribunal") if isinstance(source.get("tribunal"), dict) else {}
    return {
        "numero_processo": source.get("numeroProcesso"),
        "tribunal": tribunal.get("nome") or source.get("tribunal") or None,
        "classe": classe.get("nome") if isinstance(classe, dict) else None,
        "assuntos": _names(source.get("assuntos")),
        "data_ajuizamento": source.get("dataAjuizamento"),
        "movimentos": _movements(source.get("movimentos")),
    }


def _parse_datajud_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        if re.fullmatch(r"\d{14}", raw):
            return datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        if re.fullmatch(r"\d{8}", raw):
            return datetime.strptime(raw, "%Y%m%d").replace(tzinfo=timezone.utc)
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _percentile(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _calculate_stage_statistics(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Resumo exploratório: distribuição até o primeiro registro TPU 12740."""
    unique_cases: Dict[str, Dict[str, Any]] = {}
    duplicates_ignored = 0
    for case in cases:
        process_number = case.get("numero_processo")
        dedupe_key = str(process_number) if process_number else f"sem-numero-{len(unique_cases)}"
        if dedupe_key in unique_cases:
            duplicates_ignored += 1
            continue
        unique_cases[dedupe_key] = case

    durations_days: List[float] = []
    without_distribution = 0
    without_hearing_record = 0
    for case in unique_cases.values():
        start = _parse_datajud_datetime(case.get("data_ajuizamento"))
        if start is None:
            distribution_dates = [
                _parse_datajud_datetime(movement.get("data_hora"))
                for movement in case.get("movimentos", [])
                if movement.get("codigo") == 26
            ]
            valid_distribution_dates = [date for date in distribution_dates if date is not None]
            start = min(valid_distribution_dates) if valid_distribution_dates else None
        if start is None:
            without_distribution += 1
            continue

        hearing_dates = [
            _parse_datajud_datetime(movement.get("data_hora"))
            for movement in case.get("movimentos", [])
            if movement.get("codigo") == 12740
        ]
        valid_hearing_dates = [date for date in hearing_dates if date is not None and date >= start]
        if not valid_hearing_dates:
            without_hearing_record += 1
            continue
        duration = (min(valid_hearing_dates) - start).total_seconds() / 86400
        durations_days.append(max(duration, 0.0))

    return {
        "etapa": "Distribuição até o primeiro registro de audiência de conciliação",
        "codigo_movimento_fim": 12740,
        "casos_com_duracao": len(durations_days),
        "processos_sem_data_de_distribuicao": without_distribution,
        "processos_sem_registro_da_etapa": without_hearing_record,
        "duplicatas_por_numero_processual_ignoradas": duplicates_ignored,
        "mediana_dias": statistics.median(durations_days) if durations_days else None,
        "p25_dias": _percentile(durations_days, 0.25),
        "p75_dias": _percentile(durations_days, 0.75),
        "interpretacao": "Estatística exploratória da amostra retornada; o movimento 12740 indica registro de audiência de conciliação, não confirma por si só que a audiência ocorreu.",
    }


def _persistent_cache_path() -> Path:
    configured = os.getenv("DATAJUD_CACHE_DB", "").strip()
    path = Path(configured) if configured else Path(__file__).resolve().parent / ".datajud_cache.sqlite3"
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    return path


def _persistent_cache_ttl() -> int:
    try:
        return max(60, min(int(os.getenv("DATAJUD_CACHE_TTL_SECONDS", str(PERSISTENT_CACHE_DEFAULT_TTL_SECONDS))), 604800))
    except ValueError:
        return PERSISTENT_CACHE_DEFAULT_TTL_SECONDS


def _cache_database() -> sqlite3.Connection:
    path = _persistent_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS datajud_stat_cache (
            cache_key text PRIMARY KEY,
            tribunal_alias text NOT NULL,
            consulted_at text NOT NULL,
            expires_at real NOT NULL,
            sample_count integer NOT NULL,
            statistics_json text NOT NULL
        )
        """
    )
    return conn


def _read_persistent_summary(cache_key: str) -> Optional[Dict[str, Any]]:
    conn = None
    try:
        conn = _cache_database()
        now = time.time()
        conn.execute("DELETE FROM datajud_stat_cache WHERE expires_at <= ?", (now,))
        row = conn.execute(
            "SELECT consulted_at, sample_count, statistics_json FROM datajud_stat_cache WHERE cache_key = ? AND expires_at > ?",
            (cache_key, now),
        ).fetchone()
        if not row:
            return None
        return {
            "consultado_em": row[0],
            "amostras_retornadas": int(row[1]),
            "estatistica_etapa": json.loads(row[2]),
        }
    except (sqlite3.Error, json.JSONDecodeError, OSError):
        return None
    finally:
        if conn:
            conn.close()


def _write_persistent_summary(
    *, cache_key: str, tribunal: str, consulted_at: str, sample_count: int, statistics_summary: Dict[str, Any]
) -> None:
    # A persistência contém apenas estatísticas agregadas, sem número de processo ou movimentos.
    if int(statistics_summary.get("casos_com_duracao", 0)) < 5:
        return
    conn = None
    try:
        conn = _cache_database()
        conn.execute(
            """
            INSERT INTO datajud_stat_cache
                (cache_key, tribunal_alias, consulted_at, expires_at, sample_count, statistics_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                tribunal_alias = excluded.tribunal_alias,
                consulted_at = excluded.consulted_at,
                expires_at = excluded.expires_at,
                sample_count = excluded.sample_count,
                statistics_json = excluded.statistics_json
            """,
            (
                cache_key,
                tribunal,
                consulted_at,
                time.time() + _persistent_cache_ttl(),
                sample_count,
                json.dumps(statistics_summary, ensure_ascii=False),
            ),
        )
        conn.commit()
    except (sqlite3.Error, OSError):
        # Falha do cache nunca deve impedir a consulta normal ao DataJud.
        return
    finally:
        if conn:
            conn.close()


def search_comparable_cases(*, tribunal: str, query: str, limit: int = 25) -> Dict[str, Any]:
    global _RATE_LIMIT_UNTIL
    alias = tribunal.strip().lower()
    if not ALIAS_RE.fullmatch(alias):
        raise DataJudError("Alias de tribunal inválido. Use o formato api_publica_tjrr.")

    query_text = " ".join(query.split())
    phrases = [" ".join(part.split()) for part in query_text.split(";") if part.strip()]
    if not phrases or any(len(phrase) < 3 for phrase in phrases) or len(query_text) > 160:
        raise DataJudError("Informe uma ou mais expressões de assunto, separadas por ponto e vírgula (3 a 160 caracteres no total).")

    cache_key = ("stage-v1", alias, query_text.casefold(), max(1, min(limit, 10)))
    persistent_key = sha256("\0".join(map(str, cache_key)).encode("utf-8")).hexdigest()
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _SEARCH_CACHE.get(cache_key)
        if cached and cached[0] > now:
            result = deepcopy(cached[1])
            result["cache_hit"] = True
            result["cache_persistente"] = False
            result["casos"] = []
            result["aviso"] = "Estatísticas agregadas reutilizadas do cache em memória. Os registros individuais não foram guardados."
            return result
        if cached:
            _SEARCH_CACHE.pop(cache_key, None)
        cooldown_remaining = int(_RATE_LIMIT_UNTIL - now)
    persistent_cached = _read_persistent_summary(persistent_key)
    if persistent_cached:
        return {
            "fonte": "DataJud/CNJ",
            "tribunal_alias": alias,
            "termos_busca": query_text,
            **persistent_cached,
            "cache_hit": True,
            "cache_persistente": True,
            "casos": [],
            "aviso": "Estatísticas agregadas reutilizadas do cache local. Os registros individuais não foram persistidos; os dados podem estar desatualizados.",
        }
    if cooldown_remaining > 0:
        raise DataJudError(
            f"O DataJud pediu uma pausa. Aguarde cerca de {cooldown_remaining} segundos antes de tentar de novo.",
            status_code=429,
            retry_after=cooldown_remaining,
        )

    api_key = os.getenv("DATAJUD_API_KEY", "").strip()
    if not api_key:
        raise DataJudError("DataJud não configurado. Defina DATAJUD_API_KEY no ambiente do backend.")

    body = {
        "size": max(1, min(limit, 10)),
        "track_total_hits": False,
        "_source": ["numeroProcesso", "tribunal", "classe", "assuntos", "dataAjuizamento", "movimentos"],
        "query": {
            "bool": {
                "filter": [{"match": {"classe.nome": "Cível"}}],
                "should": [
                    {"match_phrase": {"assuntos.nome": phrase}}
                    for phrase in phrases
                ],
                "minimum_should_match": 1,
            }
        },
    }
    request = Request(
        f"{DATAJUD_BASE_URL}/{alias}/_search",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"APIKey {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise DataJudError(
                "DataJud recusou a autenticação. Confira a chave vigente na documentação do CNJ.",
                status_code=exc.code,
            ) from exc
        if exc.code == 429:
            retry_after = 60
            retry_header = exc.headers.get("Retry-After") if exc.headers else None
            if retry_header:
                try:
                    retry_after = max(1, min(int(retry_header), 3600))
                except ValueError:
                    try:
                        retry_date = parsedate_to_datetime(retry_header)
                        retry_after = max(1, min(int((retry_date - datetime.now(timezone.utc)).total_seconds()), 3600))
                    except (TypeError, ValueError):
                        pass
            with _CACHE_LOCK:
                _RATE_LIMIT_UNTIL = time.monotonic() + retry_after
            raise DataJudError(
                f"Limite de consultas do DataJud atingido. Aguarde {retry_after} segundos antes de tentar novamente.",
                status_code=429,
                retry_after=retry_after,
            ) from exc
        raise DataJudError(f"DataJud retornou HTTP {exc.code}.") from exc
    except TimeoutError as exc:
        raise DataJudError("A consulta ao DataJud excedeu 120 segundos. Tente novamente.") from exc
    except URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise DataJudError("A consulta ao DataJud excedeu 120 segundos. Tente novamente.") from exc
        raise DataJudError("Não foi possível conectar ao DataJud agora.") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DataJudError("DataJud retornou uma resposta inválida.") from exc

    hits_obj = payload.get("hits", {}) if isinstance(payload, dict) else {}
    raw_hits = hits_obj.get("hits", []) if isinstance(hits_obj, dict) else []
    records = [_normalize_hit(hit) for hit in raw_hits if isinstance(hit, dict)]
    result = {
        "fonte": "DataJud/CNJ",
        "tribunal_alias": alias,
        "termos_busca": query_text,
        "consultado_em": datetime.now(timezone.utc).isoformat(),
        "amostras_retornadas": len(records),
        "cache_hit": False,
        "cache_persistente": False,
        "estatistica_etapa": _calculate_stage_statistics(records),
        "casos": records,
        "aviso": "Resultados públicos para exploração. A busca textual não confirma comparabilidade jurídica nem representa todos os processos do tribunal.",
    }
    _write_persistent_summary(
        cache_key=persistent_key,
        tribunal=alias,
        consulted_at=result["consultado_em"],
        sample_count=len(records),
        statistics_summary=result["estatistica_etapa"],
    )
    with _CACHE_LOCK:
        if len(_SEARCH_CACHE) >= 128:
            oldest_key = next(iter(_SEARCH_CACHE))
            _SEARCH_CACHE.pop(oldest_key, None)
        memory_summary = {key: value for key, value in result.items() if key != "casos"}
        memory_summary["casos"] = []
        _SEARCH_CACHE[cache_key] = (time.monotonic() + CACHE_TTL_SECONDS, deepcopy(memory_summary))
    return result
