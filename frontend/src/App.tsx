import { useCallback, useState } from "react";
import {
  extractPdf,
  predictFromMetadados,
  type ExtractResponse,
  type PredictResponse,
} from "./api";

/**
 * UI mínima do MVP: upload de PDF, exibição do JSON de metadados e resultado da predição.
 * Fluxo: extrair primeiro (`POST /api/extract`), depois prever com os mesmos metadados (`POST /api/predict` em JSON).
 */

function formatJson(data: unknown): string {
  return JSON.stringify(data, null, 2);
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [extractResult, setExtractResult] = useState<ExtractResponse | null>(null);
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);
  const [loadingExtract, setLoadingExtract] = useState(false);
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onPickFile = useCallback((list: FileList | null) => {
    const f = list?.[0];
    setError(null);
    setExtractResult(null);
    setPredictResult(null);
    setFile(f && f.size > 0 ? f : null);
  }, []);

  const runExtract = useCallback(async () => {
    if (!file) {
      setError("Selecione um arquivo PDF antes de extrair.");
      return;
    }
    setError(null);
    setLoadingExtract(true);
    try {
      const res = await extractPdf(file);
      setExtractResult(res);
      setPredictResult(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setLoadingExtract(false);
    }
  }, [file]);

  const runPredict = useCallback(async () => {
    if (!extractResult) {
      setError("Extraia os metadados do PDF antes de prever.");
      return;
    }
    setError(null);
    setLoadingPredict(true);
    try {
      const res = await predictFromMetadados(extractResult.metadados);
      setPredictResult(res);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setLoadingPredict(false);
    }
  }, [extractResult]);

  const hasUploaded = Boolean(file);
  const hasExtracted = Boolean(extractResult);
  const hasPredicted = Boolean(predictResult);

  return (
    <div className="app">
      <header className="hero panel">
        <p className="hero-kicker">MVP</p>
        <h1>IA Jurídica</h1>
        <p className="subtitle hero-subtitle">
          Envie um processo em PDF para extrair metadados (JSON) e estimar dias previstos.
        </p>
        <ol className="stepper" aria-label="Etapas do fluxo">
          <li
            className={`step ${hasUploaded ? "is-done" : "is-active"}`}
            aria-current={!hasUploaded ? "step" : undefined}
          >
            <span className="step-index">1</span>
            <span className="step-label">Upload</span>
          </li>
          <li
            className={`step ${hasExtracted ? "is-done" : hasUploaded ? "is-active" : "is-pending"}`}
            aria-current={hasUploaded && !hasExtracted ? "step" : undefined}
          >
            <span className="step-index">2</span>
            <span className="step-label">Metadados</span>
          </li>
          <li
            className={`step ${hasPredicted ? "is-done" : hasExtracted ? "is-active" : "is-pending"}`}
            aria-current={hasExtracted && !hasPredicted ? "step" : undefined}
          >
            <span className="step-index">3</span>
            <span className="step-label">Predição</span>
          </li>
        </ol>
      </header>

      <section className="panel">
        <h2>1. Upload</h2>
        <div className="row">
          <input
            id="pdf-file-input"
            className="file-input-hidden"
            type="file"
            accept="application/pdf,.pdf"
            onChange={(ev) => onPickFile(ev.target.files)}
          />
          <label htmlFor="pdf-file-input" className="upload-zone">
            <span className="upload-zone-title">Arraste ou clique para selecionar um PDF</span>
            <span className="upload-zone-subtitle">
              {file ? `Arquivo selecionado: ${file.name}` : "Formatos aceitos: .pdf"}
            </span>
          </label>
        </div>
        <div className="row" style={{ marginTop: "0.75rem" }}>
          <button type="button" onClick={runExtract} disabled={!file || loadingExtract}>
            {loadingExtract ? "Extraindo…" : "Extrair metadados"}
          </button>
          <button
            type="button"
            className="secondary"
            onClick={runPredict}
            disabled={!extractResult || loadingPredict}
          >
            {loadingPredict ? "Prevendo…" : "Prever dias"}
          </button>
        </div>
        {error ? <div className="error">{error}</div> : null}
      </section>

      <section className="panel">
        <h2>2. Metadados (resposta de /api/extract)</h2>
        {extractResult ? (
          <div className="json-block">
            <div className="json-block-header">
              <span className="json-block-title">JSON</span>
            </div>
            <pre className="json">{formatJson(extractResult)}</pre>
          </div>
        ) : (
          <p className="subtitle" style={{ margin: 0 }}>
            Nenhum resultado ainda. Selecione um PDF e clique em &quot;Extrair metadados&quot;.
          </p>
        )}
      </section>

      <section className="panel">
        <h2>3. Predição (resposta de /api/predict)</h2>
        {predictResult ? (
          <>
            <div className="prediction-card">
              <p className="prediction-label">Dias previstos</p>
              <p className="prediction-highlight">
                <span className="prediction-value">
                  {Number.isInteger(predictResult.dias_previstos)
                    ? predictResult.dias_previstos
                    : predictResult.dias_previstos.toFixed(2)}
                </span>{" "}
                <span className="prediction-unit">dias</span>
              </p>
            </div>
            <h3 className="section-title">Casos similares</h3>
            {predictResult.casos_similares.length === 0 ? (
              <p className="subtitle" style={{ margin: 0 }}>
                Nenhum caso similar retornado.
              </p>
            ) : (
              <div className="table-wrap">
                <table className="table-similares">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Dias totais</th>
                    </tr>
                  </thead>
                  <tbody>
                    {predictResult.casos_similares.map((c) => (
                      <tr key={c.id}>
                        <td className="similar-id">{c.id}</td>
                        <td>{c.dias_totais}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <details className="json-details">
              <summary>JSON completo</summary>
              <div className="json-block">
                <div className="json-block-header">
                  <span className="json-block-title">JSON</span>
                </div>
                <pre className="json">{formatJson(predictResult)}</pre>
              </div>
            </details>
          </>
        ) : (
          <p className="subtitle" style={{ margin: 0 }}>
            Após a extração, clique em &quot;Prever dias&quot; para ver a estimativa e casos
            similares.
          </p>
        )}
      </section>

      <p className="hint">
        API padrão: <code>http://127.0.0.1:8000</code>. Para outro host, defina{" "}
        <code>VITE_API_BASE</code> ao rodar o Vite.
      </p>
    </div>
  );
}
