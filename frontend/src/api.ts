/**
 * Tipos e chamadas HTTP alinhados ao contrato do FastAPI (`backend/main.py`).
 *
 * Por que separar: mantém o componente de UI focado em estado e interação,
 * e centraliza a URL base e o tratamento de erro HTTP.
 */

export type Partes = {
  autor: string | null;
  reu: string | null;
};

export type MetadadosExtracted = {
  numero_processo: string | null;
  valor_causa: number | null;
  comarca: string | null;
  juiz: string | null;
  assunto: string | null;
  partes: Partes;
};

export type ExtractResponse = {
  metadados: MetadadosExtracted;
  gerado_em: string;
};

export type CasoSimilar = {
  id: string;
  dias_totais: number;
};

export type PredictResponse = {
  dias_previstos: number;
  casos_similares: CasoSimilar[];
  gerado_em: string;
};

function apiBase(): string {
  const base = import.meta.env.VITE_API_BASE?.trim();
  if (base) {
    return base.replace(/\/$/, "");
  }
  return "http://127.0.0.1:8000";
}

async function readErrorDetail(res: Response): Promise<string> {
  try {
    const data: unknown = await res.json();
    if (
      data &&
      typeof data === "object" &&
      "detail" in data &&
      typeof (data as { detail: unknown }).detail === "string"
    ) {
      return (data as { detail: string }).detail;
    }
  } catch {
    /* ignora — abaixo usamos texto bruto */
  }
  try {
    return await res.text();
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

export async function extractPdf(file: File): Promise<ExtractResponse> {
  const fd = new FormData();
  fd.append("file", file);

  const url = `${apiBase()}/api/extract`;
  const res = await fetch(url, {
    method: "POST",
    body: fd,
  });

  if (!res.ok) {
    const detail = await readErrorDetail(res);
    throw new Error(detail || `Falha na extração (${res.status}).`);
  }

  return (await res.json()) as ExtractResponse;
}

export async function predictFromMetadados(
  metadados: MetadadosExtracted,
): Promise<PredictResponse> {
  const url = `${apiBase()}/api/predict`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ metadados }),
  });

  if (!res.ok) {
    const detail = await readErrorDetail(res);
    throw new Error(detail || `Falha na predição (${res.status}).`);
  }

  return (await res.json()) as PredictResponse;
}
