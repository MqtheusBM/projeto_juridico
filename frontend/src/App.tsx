import { useCallback, useEffect, useState } from "react";
import {
  ApiResponseError,
  extractPdf,
  predictFromMetadados,
  searchDataJudCases,
  type ExtractResponse,
  type PredictResponse,
  type DataJudSearchResponse,
} from "./api";

/**
 * UI mínima do MVP: upload de PDF, exibição do JSON de metadados e resultado da predição.
 * Fluxo: extrair primeiro (`POST /api/extract`), depois prever com os mesmos metadados (`POST /api/predict` em JSON).
 */

function formatJson(data: unknown): string {
  return JSON.stringify(data, null, 2);
}

function formatDataJudDate(value: string | null): string {
  if (!value) return "data não informada";
  const compact = value.match(/^(\d{4})(\d{2})(\d{2})/);
  if (compact) return `${compact[3]}/${compact[2]}/${compact[1]}`;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString("pt-BR");
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [extractResult, setExtractResult] = useState<ExtractResponse | null>(null);
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);
  const [loadingExtract, setLoadingExtract] = useState(false);
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [datajudError, setDatajudError] = useState<string | null>(null);
  const [datajudTribunal, setDatajudTribunal] = useState("api_publica_tjrr");
  const [datajudTerms, setDatajudTerms] = useState("Extravio de bagagem; Transporte Aéreo");
  const [datajudResult, setDatajudResult] = useState<DataJudSearchResponse | null>(null);
  const [loadingDatajud, setLoadingDatajud] = useState(false);
  const [datajudRetrySeconds, setDatajudRetrySeconds] = useState(0);
  const waitingForDataJud = datajudRetrySeconds > 0;

  useEffect(() => {
    if (!waitingForDataJud) return;
    const timer = window.setInterval(() => {
      setDatajudRetrySeconds((seconds) => Math.max(0, seconds - 1));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [waitingForDataJud]);

  const runDatajudSearch = useCallback(async () => {
    setDatajudError(null);
    setLoadingDatajud(true);
    try {
      setDatajudResult(await searchDataJudCases({
        tribunal: datajudTribunal,
        termos: datajudTerms,
        limite: 10,
      }));
    } catch (e) {
      setDatajudError(e instanceof Error ? e.message : String(e));
      if (e instanceof ApiResponseError && e.status === 429) {
        setDatajudRetrySeconds(e.retryAfterSeconds ?? 60);
      }
      setDatajudResult(null);
    } finally {
      setLoadingDatajud(false);
    }
  }, [datajudTribunal, datajudTerms]);

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
        <h2>Pesquisa exploratória de processos públicos</h2>
        <p className="subtitle">
          Consulte processos cíveis do DataJud por assuntos processuais. Separe expressões alternativas com ponto e vírgula. A chave fica somente no backend.
        </p>
        <div className="datajud-form">
          <label>
            Alias do tribunal
            <input value={datajudTribunal} onChange={(event) => setDatajudTribunal(event.target.value)} />
          </label>
          <label>
            Assunto ou termos
            <input value={datajudTerms} onChange={(event) => setDatajudTerms(event.target.value)} />
          </label>
          <button type="button" onClick={runDatajudSearch} disabled={loadingDatajud || waitingForDataJud || datajudTerms.trim().length < 3}>
            {loadingDatajud ? "Consultando…" : waitingForDataJud ? `Tente em ${datajudRetrySeconds}s` : "Buscar no DataJud"}
          </button>
        </div>
        {datajudError ? (
          <div className="error" role="alert">
            {datajudError}
            {waitingForDataJud ? <div>Nova tentativa disponível em {datajudRetrySeconds} segundos.</div> : null}
          </div>
        ) : null}
        {datajudResult ? (
          <>
            <p className="datajud-summary">
              {datajudResult.amostras_retornadas} amostras retornadas · consulta em {new Date(datajudResult.consultado_em).toLocaleString("pt-BR")}
              {datajudResult.cache_hit
                ? datajudResult.cache_persistente
                  ? " · estatística agregada reutilizada do cache persistente"
                  : " · resultado reutilizado do cache em memória"
                : ""}
            </p>
            <p className="datajud-notice">{datajudResult.aviso}</p>
            {datajudResult.cache_hit && datajudResult.casos.length === 0 ? (
              <p className="datajud-notice">Esta resposta reutiliza apenas estatísticas agregadas; os detalhes individuais dos processos não foram guardados.</p>
            ) : null}
            <section className="stage-stat" aria-label="Estatística exploratória por etapa">
              <h3>{datajudResult.estatistica_etapa.etapa}</h3>
              {datajudResult.estatistica_etapa.mediana_dias !== null ? (
                <p>
                  Em {datajudResult.estatistica_etapa.casos_com_duracao} processo(s), a mediana foi{" "}
                  <strong>{datajudResult.estatistica_etapa.mediana_dias.toFixed(1)} dias</strong>
                  {datajudResult.estatistica_etapa.p25_dias !== null && datajudResult.estatistica_etapa.p75_dias !== null
                    ? "; P25–P75: " + datajudResult.estatistica_etapa.p25_dias.toFixed(1) + "–" + datajudResult.estatistica_etapa.p75_dias.toFixed(1) + " dias."
                    : "."}
                </p>
              ) : (
                <p>Não há processos suficientes com as duas datas para calcular esta etapa na amostra.</p>
              )}
              <p className="datajud-notice">
                {datajudResult.estatistica_etapa.interpretacao} Sem registro da etapa: {datajudResult.estatistica_etapa.processos_sem_registro_da_etapa};
                {" "}sem data de distribuição: {datajudResult.estatistica_etapa.processos_sem_data_de_distribuicao};
                {" "}duplicatas ignoradas: {datajudResult.estatistica_etapa.duplicatas_por_numero_processual_ignoradas}.
              </p>
            </section>
            <div className="datajud-cases">
              {datajudResult.casos.map((caso, index) => (
                <article className="datajud-case" key={`${caso.numero_processo ?? "caso"}-${index}`}>
                  <strong>{caso.numero_processo ?? "Número não informado"}</strong>
                  <span>{caso.tribunal ?? datajudResult.tribunal_alias} · {caso.classe ?? "Classe não informada"}</span>
                  <span>Assuntos: {caso.assuntos.join(", ") || "não informados"}</span>
                  <span>Distribuição: {formatDataJudDate(caso.data_ajuizamento)} · {caso.movimentos.length} movimentações</span>
                </article>
              ))}
            </div>
          </>
        ) : null}
      </section>

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
