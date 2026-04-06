"""
Tarefas Celery: OCR/extração e predição em processo separado (V1).

Por que separar do FastAPI:
- PyMuPDF/Tesseract/embeddings podem ser lentos; a API responde rápido com `task_id`.
- O worker executa `bootstrap_backend_runtime()` para ter PostgreSQL/XGBoost como na API.

Causa comum de falha no Windows:
- Pool `prefork` não funciona bem; use `--pool=solo` ao iniciar o worker.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from celery.signals import worker_process_init

from audit_log import audit_event, safe_metadados_summary
from celery_app import celery_app

logger = logging.getLogger("juridico-celery")


@worker_process_init.connect
def _carregar_env_e_bootstrap(**_kwargs: Any) -> None:
    """
    No startup de cada processo worker: carrega `.env` na raiz do repositório e inicializa DB/ML.

    Por que aqui: o worker não passa pelos eventos `startup` do FastAPI; sem isso, `DB_ENGINE` ficaria vazio.
    """

    try:
        from pathlib import Path

        from dotenv import load_dotenv

        raiz_repo = Path(__file__).resolve().parent.parent
        load_dotenv(raiz_repo / ".env")
    except Exception as e:
        logger.warning("Não foi possível carregar .env no worker. (causa: %s)", type(e).__name__)

    # Import tardio evita ciclo com `main` durante o import de `tasks`.
    from main import bootstrap_backend_runtime

    bootstrap_backend_runtime()
    audit_event("celery_worker_ready", pid=os.getpid())
    logger.info("Worker Celery inicializado (persistência/ML conforme ambiente).")


@celery_app.task(bind=True, name="juridico.extract_pdf")
def extract_pdf_from_path_task(self, pdf_path: str) -> Dict[str, Any]:
    """
    Lê PDF do caminho temporário, extrai metadados e apaga o arquivo.

    Parâmetros:
      pdf_path: caminho absoluto criado pela API (arquivo temporário).

    Retorno:
      Dicionário JSON-serializável compatível com `ExtractResponse`.
    """

    from main import run_extract_from_pdf_bytes

    audit_event("celery_extract_enqueued", task_id=self.request.id)
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        audit_event(
            "celery_extract_started",
            task_id=self.request.id,
            pdf_bytes=len(pdf_bytes),
        )
        result = run_extract_from_pdf_bytes(pdf_bytes)
        audit_event(
            "celery_extract_finished",
            task_id=self.request.id,
            **safe_metadados_summary(result.metadados),
        )
        return result.model_dump(mode="json")
    finally:
        try:
            os.unlink(pdf_path)
        except OSError:
            logger.warning("Não foi possível remover arquivo temporário do extract.")


@celery_app.task(bind=True, name="juridico.predict_pdf")
def predict_from_pdf_path_task(self, pdf_path: str) -> Dict[str, Any]:
    """
    Pipeline completo em background: PDF -> texto -> metadados -> RAG/predição.

    Parâmetros:
      pdf_path: arquivo temporário enviado pelo cliente.

    Retorno:
      Dicionário compatível com `PredictResponse`.
    """

    from main import run_predict_from_pdf_bytes

    audit_event("celery_predict_pdf_enqueued", task_id=self.request.id)
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        audit_event(
            "celery_predict_pdf_started",
            task_id=self.request.id,
            pdf_bytes=len(pdf_bytes),
        )
        result = run_predict_from_pdf_bytes(pdf_bytes)
        audit_event(
            "celery_predict_pdf_finished",
            task_id=self.request.id,
            dias_previstos=result.dias_previstos,
            casos_similares_count=len(result.casos_similares),
        )
        return result.model_dump(mode="json")
    finally:
        try:
            os.unlink(pdf_path)
        except OSError:
            logger.warning("Não foi possível remover arquivo temporário do predict.")


@celery_app.task(bind=True, name="juridico.predict_metadados")
def predict_from_metadados_task(self, metadados_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predição a partir de JSON de metadados (sem PDF).

    Parâmetros:
      metadados_dict: payload já no formato esperado por `MetadadosExtracted`.
    """

    from main import MetadadosExtracted, run_predict_core

    audit_event("celery_predict_meta_enqueued", task_id=self.request.id)
    meta = MetadadosExtracted.model_validate(metadados_dict)
    audit_event(
        "celery_predict_meta_started",
        task_id=self.request.id,
        **safe_metadados_summary(meta),
    )
    result = run_predict_core(meta, texto_para_embedding=None)
    audit_event(
        "celery_predict_meta_finished",
        task_id=self.request.id,
        dias_previstos=result.dias_previstos,
        casos_similares_count=len(result.casos_similares),
    )
    return result.model_dump(mode="json")
