"""
Aplicação Celery (broker Redis) para tarefas pesadas em background (V1).

Por que Redis:
- Broker e backend de resultados simples para desenvolvimento local e pequenos deploys.

Como executar o worker (Windows costuma exigir pool single-thread):
  cd backend
  celery -A celery_app worker --loglevel=info --pool=solo

Variáveis:
- CELERY_BROKER_URL (default redis://localhost:6379/0)
- CELERY_RESULT_BACKEND (opcional; se vazio, reutiliza o broker)
"""

from __future__ import annotations

import os

from celery import Celery


def _broker_url() -> str:
    return os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0").strip()


def _result_backend() -> str:
    explicit = os.getenv("CELERY_RESULT_BACKEND", "").strip()
    return explicit if explicit else _broker_url()


celery_app = Celery(
    "juridico",
    broker=_broker_url(),
    backend=_result_backend(),
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)


def _register_task_modules() -> None:
    """
    Garante que o worker registre as tasks ao importar `celery_app`.

    Por que em função: o `uvicorn` carrega só `main` e não precisa registrar tasks até
    alguém chamar o endpoint assíncrono (import lazy em `main`).
    """

    import tasks  # noqa: F401


_register_task_modules()
