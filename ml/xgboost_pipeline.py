from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np  # type: ignore[import-not-found]


class SafeLabelEncoder:
    """
    LabelEncoder "seguro" para inferência.

    Por que existe:
    - O sklearn `LabelEncoder` não lida bem com valores novos (categorias não vistas).
    - No nosso caso, comarca/juiz/assunto podem aparecer novos em tempo de execução.

    Comportamento:
    - valores desconhecidos viram `unknown_value` (por padrão: -1)
    - valores vazios/None viram uma categoria "MISSING" (você decide se quer pré-tratar)
    """

    def __init__(self, *, unknown_value: int = -1):
        self.unknown_value = int(unknown_value)
        self.mapping: Dict[str, int] = {}

    def fit(self, values: Iterable[Any]) -> "SafeLabelEncoder":
        """
        Aprende um mapeamento determinístico categoria -> inteiro.

        Parametros:
          values: iterável com valores categóricos.
        Retorno:
          self (para encadeamento).
        """
        uniq: List[str] = []
        seen = set()
        for v in values:
            s = str(v).strip() if v is not None else ""
            if not s:
                s = "MISSING"
            if s not in seen:
                uniq.append(s)
                seen.add(s)
        # Ordenar garante determinismo entre execuções (importante para reprodutibilidade).
        uniq_sorted = sorted(uniq)
        self.mapping = {cat: i for i, cat in enumerate(uniq_sorted)}
        return self

    def transform(self, values: Iterable[Any]) -> np.ndarray:
        """
        Transforma categorias em inteiros, usando `unknown_value` para valores novos.

        Parametros:
          values: iterável com valores categóricos.
        Retorno:
          array numpy int64 com mesmo tamanho do input.
        """
        out: List[int] = []
        for v in values:
            s = str(v).strip() if v is not None else ""
            if not s:
                s = "MISSING"
            out.append(self.mapping.get(s, self.unknown_value))
        return np.asarray(out, dtype=np.int64)

    def fit_transform(self, values: Iterable[Any]) -> np.ndarray:
        """
        Combina fit+transform para convenience.
        """
        self.fit(values)
        return self.transform(values)


@dataclass
class XGBoostBundle:
    """
    Estrutura do artefato salvo.

    Obs: a classe não é usada obrigatoriamente na inferência agora, mas ajuda a
    manter consistência e documentar o conteúdo do joblib.
    """

    model: Any
    comarca_encoder: SafeLabelEncoder
    juiz_encoder: SafeLabelEncoder
    assunto_encoder: SafeLabelEncoder
    valor_causa_scaler: Any
    feature_columns: List[str]
    mae_test: Optional[float]
    training_config: Dict[str, Any]

