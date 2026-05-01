import React, { useEffect, useMemo, useState } from "react";
import { loadModel, predictPokemon } from "./ml/predict.js";
import { isMobileLike } from "./utils/device.js";

const SAMPLE_PATH = "/sample/test1.jpg";

export default function App() {
  const [isBlocked, setIsBlocked] = useState(false);
  const [modelStatus, setModelStatus] = useState("loading");
  const [previewUrl, setPreviewUrl] = useState(SAMPLE_PATH);
  const [fileName, setFileName] = useState("test1.jpg");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isPredicting, setIsPredicting] = useState(false);

  const topPrediction = result?.predictions?.[0];
  const statusText = useMemo(() => {
    if (modelStatus === "ready") return "Model ready";
    if (modelStatus === "error") return "Model failed";
    return "Loading model";
  }, [modelStatus]);

  useEffect(() => {
    setIsBlocked(isMobileLike());
    loadModel()
      .then(() => setModelStatus("ready"))
      .catch((loadError) => {
        setModelStatus("error");
        setError(loadError.message);
      });
  }, []);

  async function handleFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Choose an image file.");
      return;
    }

    setResult(null);
    setError("");
    setFileName(file.name);
    setPreviewUrl((previousUrl) => {
      if (previousUrl?.startsWith("blob:")) URL.revokeObjectURL(previousUrl);
      return URL.createObjectURL(file);
    });
    await runPrediction(file);
  }

  async function handleSample() {
    setResult(null);
    setError("");
    setFileName("test1.jpg");
    setPreviewUrl(SAMPLE_PATH);

    const response = await fetch(SAMPLE_PATH);
    const blob = await response.blob();
    const file = new File([blob], "test1.jpg", { type: blob.type || "image/jpeg" });
    await runPrediction(file);
  }

  async function runPrediction(file) {
    if (modelStatus !== "ready") return;

    setIsPredicting(true);
    try {
      setResult(await predictPokemon(file));
    } catch (predictionError) {
      setError(predictionError.message);
    } finally {
      setIsPredicting(false);
    }
  }

  if (isBlocked) {
    return (
      <main className="app-shell centered-shell">
        <section className="blocked-panel">
          <p className="kicker">Desktop browser recommended</p>
          <h1>Pokedex Classifier</h1>
          <p>This client-side model runs best on desktop Chrome or Edge.</p>
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <section className="pokedex">
        <aside className="control-deck">
          <div className="brand-block">
            <div className="lens-row" aria-hidden="true">
              <span className="lens lens-main" />
              <span className="lens lens-red" />
              <span className="lens lens-yellow" />
              <span className="lens lens-green" />
            </div>
            <p className="kicker">ONNX Runtime Web</p>
            <h1>Pokedex Classifier</h1>
          </div>

          <div className={`status-chip ${modelStatus}`}>
            <span />
            {statusText}
          </div>

          <label className="upload-zone">
            <input type="file" accept="image/*" onChange={handleFileChange} disabled={modelStatus !== "ready"} />
            <strong>Upload Pokemon Image</strong>
            <small>{fileName}</small>
          </label>

          <div className="action-row">
            <button type="button" onClick={handleSample} disabled={modelStatus !== "ready" || isPredicting}>
              Run Sample
            </button>
          </div>

          {error ? <p className="error-box">{error}</p> : null}
        </aside>

        <section className="screen-panel" aria-label="Selected image preview">
          <div className="screen-header">
            <span>Live Scan</span>
            <span>{isPredicting ? "Running" : modelStatus === "ready" ? "Standby" : "Booting"}</span>
          </div>
          <div className="preview-screen">
            <img src={previewUrl} alt="Pokemon preview" />
            {isPredicting ? <div className="scan-overlay">Classifying...</div> : null}
          </div>
        </section>

        <aside className="result-deck">
          <p className="kicker">Prediction</p>
          {topPrediction ? (
            <>
              <div className="top-result">
                <h2>{topPrediction.label}</h2>
                <p>{formatPercent(topPrediction.confidence)}</p>
              </div>
              <p className="latency">Inference time: {Math.round(result.elapsedMs)}ms</p>
              <div className="ranking-list">
                {result.predictions.map((prediction) => (
                  <div className="rank-card" key={prediction.classIndex}>
                    <span>{prediction.rank}</span>
                    <strong>{prediction.label}</strong>
                    <div className="bar-track">
                      <div style={{ width: formatPercent(prediction.confidence) }} />
                    </div>
                    <em>{formatPercent(prediction.confidence)}</em>
                  </div>
                ))}
              </div>
            </>
          ) : (
            <div className="empty-state">
              <h2>Ready to scan</h2>
              <p>Upload an image or run the sample to see the top predictions.</p>
            </div>
          )}
        </aside>
      </section>
    </main>
  );
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}
