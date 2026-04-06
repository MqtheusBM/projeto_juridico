"""
Inferência do XGBoost treinado em `ml/train_xgboost.py`.

Por que um módulo separado:
- Mantém `main.py` focado em HTTP; o joblib precisa que o pacote `ml` esteja importável
  (mesmas classes usadas no pickle), então ajustamos `sys.path` aqui de forma controlada.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import numpy as np  # type: ignore[import-not-found]


def _repo_root() -> Path:
    """Raiz do repositório (pasta que contém `backend/` e `ml/`)."""
    return Path(__file__).resolve().parents[1]


def _ensure_ml_importable() -> None:
    """
    Garante imports `ml.*` para unpickling do joblib.

    Causa típica de erro sem isso: executar uvicorn com cwd em `backend/` sem PYTHONPATH
    na raiz — o pickle referencia `ml.xgboost_pipeline.XGBoostBundle`.
    """
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_default_model_path() -> Path:
    """
    Caminho do artefato: variável `XGB_MODEL_PATH`, senão o mesmo default do treino.
    """
    raw = os.getenv("XGB_MODEL_PATH") or os.getenv("XGB_MODEL_OUTPUT")
    if raw:
        return Path(raw).expanduser()
    return _repo_root() / "ml" / "artifacts" / "xgboost_model.joblib"


def load_xgboost_bundle(model_path: Optional[Path] = None) -> tuple[Optional[Any], Optional[str]]:
    """
    Carrega o joblib do modelo.

    Retorno:
      (bundle, None) em sucesso; (None, mensagem de erro) se arquivo ausente ou load falhar.
    """
    path = model_path or resolve_default_model_path()
    if not path.is_file():
        return None, f"arquivo não encontrado: {path}"

    try:
        _ensure_ml_importable()
        import joblib  # type: ignore[import-not-found]

        bundle = joblib.load(str(path))
        return bundle, None
    except Exception as e:  # noqa: BLE001 — queremos logar qualquer falha de load
        return None, str(e)


def similaridade_rag_from_casos(
    casos_dias: List[int],
    *,
    global_train_mean_dias: Optional[float],
) -> float:
    """
    Replica a feature `similaridade_rag` do treino: média dos vizinhos ou média global do treino.

    Parametros:
      casos_dias: `dias_totais` dos processos similares (já ordenados/filtrados pelo caller).
      global_train_mean_dias: média de `y` no conjunto de treino (vem do `training_config`).

    Retorno:
      float usado como última coluna do vetor de features.
    """
    if casos_dias:
        return float(sum(casos_dias)) / float(len(casos_dias))
    if global_train_mean_dias is not None:
        return float(global_train_mean_dias)
    # Artefatos antigos sem metadata: fallback estável (só afeta a feature, não o contrato da API).
    return 365.0


def predict_dias_with_bundle(
    bundle: Any,
    *,
    comarca: Optional[str],
    juiz: Optional[str],
    assunto: Optional[str],
    valor_causa: Optional[float],
    similaridade_rag: float,
) -> float:
    """
    Uma linha de predição, alinhada a `_build_X` em `ml/train_xgboost.py`.

    Retorno:
      dias previstos (>= 1.0 após clamp).
    """
    cfg = getattr(bundle, "training_config", None) or {}
    median_valor = cfg.get("train_median_valor_causa")
    if median_valor is None:
        # Compatibilidade com bundles gravados antes desse campo; evita quebrar inferência.
        median_valor = 0.0
    median_valor = float(median_valor)

    def _cat(s: Optional[str]) -> str:
        t = (s or "").strip()
        return t if t else "MISSING"

    vc = float(valor_causa) if valor_causa is not None else median_valor

    comarca_cat = bundle.comarca_encoder.transform([_cat(comarca)])
    juiz_cat = bundle.juiz_encoder.transform([_cat(juiz)])
    assunto_cat = bundle.assunto_encoder.transform([_cat(assunto)])
    valor_scaled = bundle.valor_causa_scaler.transform(np.array([[vc]], dtype=np.float64))
    valor_scaled = valor_scaled.reshape(-1)
    sim_col = np.array([float(similaridade_rag)], dtype=np.float64)

    X = np.column_stack(
        [
            comarca_cat,
            juiz_cat,
            valor_scaled,
            assunto_cat,
            sim_col,
        ]
    )
    raw = float(bundle.model.predict(X)[0])
    return float(max(1.0, raw))
