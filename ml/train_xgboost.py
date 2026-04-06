from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _ensure_repo_root_on_syspath() -> None:
    """
    Garante que `import backend.*` funcione quando o script é executado como:
      python ml/train_xgboost.py
    """
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


_ensure_repo_root_on_syspath()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _parse_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "t"}:
        return True
    if v in {"0", "false", "no", "n", "f"}:
        return False
    return default


def _load_training_df(engine: Any, *, max_rows: Optional[int] = None) -> Any:
    """
    Carrega dataset base para treino.

    Retorno:
      pandas.DataFrame com colunas:
        - processo_id (uuid string)
        - numero_processo
        - tribunal
        - comarca
        - juiz_sentenciante
        - assunto_principal
        - valor_da_causa
        - dias_totais (target)
    """
    import pandas as pd  # type: ignore[import-not-found]  # import tardio

    sql = """
    SELECT
        ph.id AS processo_id,
        ph.numero_processo,
        ph.tribunal,
        ph.comarca,
        ph.juiz_sentenciante,
        ph.assunto_principal,
        ph.valor_da_causa,
        s.dias_totais
    FROM processos_historicos ph
    JOIN sentencas s ON s.processo_id = ph.id
    WHERE s.dias_totais IS NOT NULL
    """
    df = pd.read_sql_query(sql, engine)
    if max_rows and int(max_rows) > 0:
        df = df.head(int(max_rows)).copy()
    return df


def _compute_embedding_texts(
    df: Any,
    *,
    from_rag_module: bool = True,
) -> Any:
    """
    Cria a coluna `embedding_text` usada para consultar o Chroma.
    """
    import pandas as pd  # type: ignore[import-not-found]  # import tardio
    from backend.rag_chroma import build_embedding_text_from_sentence

    def _to_payload(row: Any) -> SimpleNamespace:
        return SimpleNamespace(
            numero_processo=row.get("numero_processo"),
            assunto_principal=row.get("assunto_principal"),
            tribunal=row.get("tribunal"),
            comarca=row.get("comarca"),
            juiz_sentenciante=row.get("juiz_sentenciante"),
            valor_da_causa=row.get("valor_da_causa"),
        )

    embedding_texts: List[str] = []
    for _, r in df.iterrows():
        payload = _to_payload(r)
        embedding_texts.append(build_embedding_text_from_sentence(payload))
    df = df.copy()
    df["embedding_text"] = embedding_texts
    return df


def _ensure_chroma_embeddings_for_training(
    df: Any,
    *,
    rebuild: bool,
) -> None:
    """
    (Opcional) Reinsere embeddings no ChromaDB para todos os processos.

    Por que isso existe:
    - `similaridade_rag` depende de busca vetorial via Chroma.
    - Em MVP, o usuário pode treinar antes de popular o Chroma.
    """
    if not rebuild:
        return

    from backend.rag_chroma import upsert_process_embedding

    # Upsert é idempotente: caso o id já exista, o Chroma atualiza o registro.
    for _, r in df.iterrows():
        processo_id = str(r["processo_id"])
        emb_text = str(r["embedding_text"] or "")
        if emb_text.strip():
            upsert_process_embedding(processo_id, emb_text, metadata={})


def _fetch_dias_totais_by_ids(engine: Any, process_ids: Sequence[str]) -> Dict[str, int]:
    """
    Busca target (dias_totais) para os IDs candidatos.
    """
    from backend.db import get_dias_totais_by_process_ids

    if not process_ids:
        return {}
    rows = get_dias_totais_by_process_ids(engine, process_ids=list(process_ids))
    return {str(r["id"]): int(r["dias_totais"]) for r in rows}


def _compute_similaridade_rag_from_chroma(
    engine: Any,
    df_subset: Any,
    *,
    allowed_train_ids: Set[str],
    global_train_mean: float,
    top_k_similares: int,
) -> Any:
    """
    Calcula `similaridade_rag` para um subconjunto (train ou test).

    Regra didática:
    - Para evitar "vazar" target do conjunto teste, filtramos candidatos para pertencerem
      ao conjunto `allowed_train_ids`.
    """
    from backend.rag_chroma import query_top_k_process_ids

    # 1) Monta candidatos por linha (IDs), já filtrando o "universo permitido".
    row_to_candidate_ids: Dict[str, List[str]] = {}
    unique_candidate_ids: Set[str] = set()

    # Para compensar remoção do próprio processo e interseção com train_ids,
    # pedimos um top_k um pouco maior.
    query_top_k = int(top_k_similares) + 5

    for _, r in df_subset.iterrows():
        processo_id = str(r["processo_id"])
        emb_text = str(r["embedding_text"] or "")
        if not emb_text.strip():
            row_to_candidate_ids[processo_id] = []
            continue

        candidate_ids = query_top_k_process_ids(emb_text, top_k=query_top_k)
        filtered: List[str] = []
        for cid in candidate_ids:
            if cid == processo_id:
                continue
            if cid not in allowed_train_ids:
                continue
            filtered.append(cid)
            if len(filtered) >= top_k_similares:
                break

        row_to_candidate_ids[processo_id] = filtered
        unique_candidate_ids.update(filtered)

    # 2) Busca o target real (dias_totais) apenas uma vez por conjunto de ids.
    dias_map = _fetch_dias_totais_by_ids(engine, sorted(unique_candidate_ids))

    # 3) Calcula a média por linha.
    similar_vals: List[float] = []
    for _, r in df_subset.iterrows():
        processo_id = str(r["processo_id"])
        ids = row_to_candidate_ids.get(processo_id) or []
        days = [dias_map[cid] for cid in ids if cid in dias_map]
        if not days:
            similar_vals.append(float(global_train_mean))
        else:
            similar_vals.append(float(sum(days)) / float(len(days)))

    df_subset = df_subset.copy()
    df_subset["similaridade_rag"] = similar_vals
    return df_subset


def _compute_similaridade_rag_from_sql_fallback(
    engine: Any,
    df_subset: Any,
    *,
    allowed_train_ids: Set[str],
    global_train_mean: float,
    top_k_similares: int,
) -> Any:
    """
    Fallback quando ChromaDB/RAG não está disponível.

    Abordagem:
    - Mantém a mesma ideia do feature: "média dos processos mais parecidos".
    - Como não temos busca vetorial, usamos aproximação por filtros de comarca/assunto/juiz
      e ordenamos por `data_sentenca` (recência/maior data).
    """
    import pandas as pd  # type: ignore[import-not-found]  # import tardio

    sql = """
    SELECT
        ph2.id AS processo_id,
        s2.dias_totais AS dias_totais
    FROM processos_historicos ph2
    JOIN sentencas s2 ON s2.processo_id = ph2.id
    WHERE s2.dias_totais IS NOT NULL
      AND ph2.id = ANY(:allowed_ids)
      AND ph2.id <> :self_id
      AND (:comarca IS NULL OR ph2.comarca ILIKE :comarca)
      AND (:assunto IS NULL OR ph2.assunto_principal ILIKE :assunto)
      AND (:juiz IS NULL OR ph2.juiz_sentenciante ILIKE :juiz)
    ORDER BY s2.data_sentenca DESC
    LIMIT :limit
    """

    similar_vals: List[float] = []

    allowed_ids_list = sorted(allowed_train_ids)
    if not allowed_ids_list:
        # Se não existe treino, não faz sentido calcular por "parecidos".
        df_subset = df_subset.copy()
        df_subset["similaridade_rag"] = [float(global_train_mean)] * len(df_subset)
        return df_subset

    with engine.begin() as conn:
        for _, r in df_subset.iterrows():
            self_id = str(r["processo_id"])
            comarca = r.get("comarca")
            assunto = r.get("assunto_principal")
            juiz = r.get("juiz_sentenciante")

            # Em MVP: tratamos filtros nulos como "ignorar".
            comarca_param = None if not comarca or str(comarca).strip() == "MISSING" else f"%{comarca}%"
            assunto_param = None if not assunto or str(assunto).strip() == "MISSING" else f"%{assunto}%"
            juiz_param = None if not juiz or str(juiz).strip() == "MISSING" else f"%{juiz}%"

            rows = conn.execute(
                sql,
                {
                    "allowed_ids": allowed_ids_list,
                    "self_id": self_id,
                    "comarca": comarca_param,
                    "assunto": assunto_param,
                    "juiz": juiz_param,
                    "limit": int(top_k_similares),
                },
            ).fetchall()
            days = [int(row[1]) for row in rows if row[1] is not None]
            if not days:
                similar_vals.append(float(global_train_mean))
            else:
                similar_vals.append(float(sum(days)) / float(len(days)))

    df_subset = df_subset.copy()
    df_subset["similaridade_rag"] = similar_vals
    return df_subset


def _train_and_save_model(
    engine: Any,
    df: Any,
    *,
    test_size: float,
    random_state: int,
    top_k_similares: int,
    model_output_path: Path,
    rebuild_chroma: bool,
) -> Dict[str, Any]:
    """
    Treina XGBoost + engenharia de features e salva artefato para inferência.
    """
    import numpy as np  # type: ignore[import-not-found]
    import pandas as pd  # type: ignore[import-not-found]
    import joblib  # type: ignore[import-not-found]
    from sklearn.metrics import mean_absolute_error  # type: ignore[import-not-found]
    from sklearn.model_selection import train_test_split  # type: ignore[import-not-found]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]
    from xgboost import XGBRegressor  # type: ignore[import-not-found]

    from ml.xgboost_pipeline import SafeLabelEncoder, XGBoostBundle

    if df.empty:
        raise RuntimeError("Dataset vazio: não há processos_historicos/sentencas para treinar.")

    # [INÍCIO: split 80/20]
    train_df, test_df = train_test_split(
        df,
        test_size=float(test_size),
        random_state=int(random_state),
        shuffle=True,
    )
    if train_df.empty or test_df.empty:
        raise RuntimeError("Split inválido: ajuste `test_size` ou verifique quantos registros existem.")
    # [FIM: split 80/20]

    allowed_train_ids: Set[str] = set(str(x) for x in train_df["processo_id"].tolist())
    global_train_mean = float(train_df["dias_totais"].mean())

    # Normalizações/encoders devem ser ajustados *somente* no treino.
    # (Reduz vazamento acidental de estatísticas.)

    # Preparar colunas categóricas (com fallback de string).
    def _fill_cat(series: Any) -> Any:
        return series.fillna("MISSING").astype(str).map(lambda s: s.strip() if s else "MISSING")

    df = df.copy()
    df["comarca"] = _fill_cat(df["comarca"])
    df["juiz_sentenciante"] = _fill_cat(df["juiz_sentenciante"])
    df["assunto_principal"] = _fill_cat(df["assunto_principal"])

    # Preparar valor da causa (numérico).
    # Observação: a mediana precisa ser do treino.
    train_median_valor_causa = float(train_df["valor_da_causa"].median())
    df["valor_da_causa"] = df["valor_da_causa"].fillna(train_median_valor_causa)

    # [IMPORTANTE] Recria `train_df`/`test_df` a partir do `df` "limpo".
    # Por que: `train_test_split` retorna subconjuntos; ao alterar colunas no `df`
    # depois do split, os subconjuntos antigos podem ficar sem as transformações.
    train_df = df[df["processo_id"].astype(str).isin(allowed_train_ids)].copy()
    test_df = df[~df["processo_id"].astype(str).isin(allowed_train_ids)].copy()

    # [INÍCIO: embeddings e feature similaridade_rag]
    # Tentamos usar Chroma; se falhar por falta de dependências/config, caímos no fallback SQL.
    rag_available = True
    try:
        import chromadb  # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        rag_available = False

    # Gera texto para embeddings (mesmo que caiamos no fallback, isso é barato).
    df = _compute_embedding_texts(df)

    # Recria novamente os subconjuntos para garantir que tenham a coluna `embedding_text`.
    train_df = df[df["processo_id"].astype(str).isin(allowed_train_ids)].copy()
    test_df = df[~df["processo_id"].astype(str).isin(allowed_train_ids)].copy()

    _maybe_rebuild = bool(rebuild_chroma)
    if rag_available:
        try:
            _ensure_chroma_embeddings_for_training(df, rebuild=_maybe_rebuild)
        except Exception:
            # Se falhar, ainda conseguimos rodar com fallback SQL.
            rag_available = False

    # Calcula a feature para train e test.
    if rag_available:
        train_df = _compute_similaridade_rag_from_chroma(
            engine,
            train_df,
            allowed_train_ids=allowed_train_ids,
            global_train_mean=global_train_mean,
            top_k_similares=top_k_similares,
        )
        test_df = _compute_similaridade_rag_from_chroma(
            engine,
            test_df,
            allowed_train_ids=allowed_train_ids,
            global_train_mean=global_train_mean,
            top_k_similares=top_k_similares,
        )
    else:
        train_df = _compute_similaridade_rag_from_sql_fallback(
            engine,
            train_df,
            allowed_train_ids=allowed_train_ids,
            global_train_mean=global_train_mean,
            top_k_similares=top_k_similares,
        )
        test_df = _compute_similaridade_rag_from_sql_fallback(
            engine,
            test_df,
            allowed_train_ids=allowed_train_ids,
            global_train_mean=global_train_mean,
            top_k_similares=top_k_similares,
        )

    # Garantir tipo numérico consistente para regressão.
    train_df["similaridade_rag"] = train_df["similaridade_rag"].astype(float)
    test_df["similaridade_rag"] = test_df["similaridade_rag"].astype(float)
    # [FIM: embeddings e feature similaridade_rag]

    # [INÍCIO: encoders (Label Encoding) + scaler (Valor da Causa)]
    comarca_encoder = SafeLabelEncoder()
    juiz_encoder = SafeLabelEncoder()
    assunto_encoder = SafeLabelEncoder()

    comarca_encoder.fit(train_df["comarca"].tolist())
    juiz_encoder.fit(train_df["juiz_sentenciante"].tolist())
    assunto_encoder.fit(train_df["assunto_principal"].tolist())

    # Escala numérica do valor da causa (normalização).
    valor_causa_scaler = StandardScaler()
    valor_causa_scaler.fit(train_df[["valor_da_causa"]].to_numpy())

    # [FIM: encoders + scaler]

    feature_columns = [
        "comarca_cat",
        "juiz_cat",
        "valor_causa",
        "assunto_cat",
        "similaridade_rag",
    ]

    def _build_X(dfx: Any) -> np.ndarray:
        comarca_cat = comarca_encoder.transform(dfx["comarca"].tolist())
        juiz_cat = juiz_encoder.transform(dfx["juiz_sentenciante"].tolist())
        assunto_cat = assunto_encoder.transform(dfx["assunto_principal"].tolist())

        valor_causa_scaled = valor_causa_scaler.transform(dfx[["valor_da_causa"]].to_numpy())
        valor_causa_scaled = valor_causa_scaled.reshape(-1)

        similaridade_rag = dfx["similaridade_rag"].astype(float).to_numpy().reshape(-1)

        X = np.column_stack(
            [
                comarca_cat,
                juiz_cat,
                valor_causa_scaled,
                assunto_cat,
                similaridade_rag,
            ]
        )
        return X

    X_train = _build_X(train_df)
    X_test = _build_X(test_df)

    y_train = train_df["dias_totais"].astype(float).to_numpy()
    y_test = test_df["dias_totais"].astype(float).to_numpy()

    # [INÍCIO: treino XGBoost]
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=int(random_state),
        tree_method=os.getenv("XGB_TREE_METHOD", "hist"),
    )

    model.fit(X_train, y_train)
    # [FIM: treino XGBoost]

    # [INÍCIO: avaliação MAE]
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    logging.getLogger("juridico-ml").info("MAE (teste): %.4f", mae)
    # [FIM: avaliação MAE]

    # [INÍCIO: persistência do artefato]
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = XGBoostBundle(
        model=model,
        comarca_encoder=comarca_encoder,
        juiz_encoder=juiz_encoder,
        assunto_encoder=assunto_encoder,
        valor_causa_scaler=valor_causa_scaler,
        feature_columns=feature_columns,
        mae_test=mae,
        training_config={
            "test_size": float(test_size),
            "random_state": int(random_state),
            "top_k_similares": int(top_k_similares),
            "rebuild_chroma": bool(rebuild_chroma),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            # Usados na inferência API: mesma lógica do treino (fillna + fallback de similaridade).
            "global_train_mean_dias": float(global_train_mean),
            "train_median_valor_causa": float(train_median_valor_causa),
        },
    )

    joblib.dump(bundle, str(model_output_path))

    # Salva um JSON humano para depuração.
    metrics_path = model_output_path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mae_test": mae,
                **bundle.training_config,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    # [FIM: persistência do artefato]

    return {
        "mae_test": mae,
        "model_output_path": str(model_output_path),
        "metrics_path": str(metrics_path),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinar pipeline XGBoost (regressão de dias).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporcao do conjunto de teste.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed para reprodutibilidade.")
    parser.add_argument("--top-k-similares", type=int, default=5, help="Quantos parecidos usar na feature RAG.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Limita linhas do dataset (0 = usa tudo). Para testes locais rápidos.",
    )
    parser.add_argument(
        "--rebuild-chroma",
        type=str,
        default=os.getenv("REBUILD_CHROMA_EMBEDDINGS", "false"),
        help="Se true, reinsere embeddings no Chroma antes de calcular similaridade_rag.",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=os.getenv("XGB_MODEL_OUTPUT", "ml/artifacts/xgboost_model.joblib"),
        help="Caminho do joblib do modelo.",
    )
    args = parser.parse_args()

    import dotenv  # type: ignore[import-not-found]

    # Carrega variáveis de ambiente da raiz quando existir `.env`.
    dotenv.load_dotenv()

    _setup_logging(os.getenv("BACKEND_LOG_LEVEL", "INFO"))
    logger = logging.getLogger("juridico-ml.train_xgboost")

    rebuild_chroma = _parse_bool(args.rebuild_chroma, default=False)
    max_rows = int(args.max_rows)
    max_rows_opt: Optional[int] = None if max_rows <= 0 else max_rows

    # [INÍCIO: bootstrap do banco]
    from backend.db import get_db_engine

    engine = get_db_engine()
    # [FIM: bootstrap do banco]

    logger.info("Carregando dataset do PostgreSQL...")
    df = _load_training_df(engine, max_rows=max_rows_opt)

    logger.info("Linhas carregadas: %s", len(df))
    if len(df) < 10:
        logger.warning(
            "Dataset pequeno (%d linhas). MAE pode ficar instável. Para MVP isso é ok, mas registre o resultado.",
            len(df),
        )

    logger.info("Treinando e salvando modelo...")
    result = _train_and_save_model(
        engine,
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        top_k_similares=args.top_k_similares,
        model_output_path=Path(args.model_output),
        rebuild_chroma=rebuild_chroma,
    )

    logger.info("Treino concluido: %s", json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

