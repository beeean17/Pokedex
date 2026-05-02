import React, { useEffect, useMemo, useState } from "react";
import { loadModel, predictPokemon } from "./ml/predict.js";

const BASE_URL = import.meta.env.BASE_URL;
const SAMPLE_PATH = `${BASE_URL}sample/test1.jpg`;
const LOW_CONFIDENCE_THRESHOLD = 0.45;
const THUMBNAIL_MANIFEST_URL = `${BASE_URL}pokemon-thumbnails/manifest.json`;
const POKEMON_KO_NAMES_URL = `${BASE_URL}i18n/pokemon-ko.json`;

const TEXT = {
  en: {
    blockedEyebrow: "Desktop browser recommended",
    blockedBody: "This client-side model runs best on desktop Chrome or Edge.",
    heroTitle: "Pokemon Image Classifier",
    heroBody: "Upload a Pokemon image and run local ONNX inference directly in the browser. No backend upload is used.",
    modelReady: "Model ready",
    modelFailed: "Model failed",
    modelLoading: "Loading model",
    chooseImage: "Choose specimen image",
    fileHelp: "PNG, JPG, WEBP. Current file:",
    runSample: "Run Sample",
    pasteUrl: "Paste URL",
    chooseImageError: "Choose an image file.",
    classifying: "Classifying",
    analysisResults: "Analysis results",
    matchesFound: "Matches found",
    readyToAnalyze: "Ready to analyze",
    matches: "matches",
    lowConfidence: "Low confidence",
    primaryMatch: "Primary match",
    topPrediction: "Top prediction",
    confidence: "confidence",
    uncertainCopy:
      "The model is uncertain. A busy background, small subject, or unusual angle may be affecting the prediction.",
    strongCopy: "The model found a strong visual match for the uploaded specimen.",
    alternativeMatch: "Alternative visual match",
    recognitionTip: "Recognition tip",
    tipBody: "For better accuracy, use a centered Pokemon with good lighting and less visual clutter in the background.",
    noAnalysis: "No analysis yet",
    noAnalysisBody: "Choose an image or run the sample to see ranked predictions here.",
  },
  ko: {
    blockedEyebrow: "데스크톱 브라우저 권장",
    blockedBody: "이 클라이언트 모델은 데스크톱 Chrome 또는 Edge에서 가장 안정적으로 동작합니다.",
    heroTitle: "포켓몬 이미지 분류기",
    heroBody: "포켓몬 이미지를 업로드하면 서버 업로드 없이 브라우저에서 바로 ONNX 추론을 실행합니다.",
    modelReady: "모델 준비 완료",
    modelFailed: "모델 로드 실패",
    modelLoading: "모델 로딩 중",
    chooseImage: "분석할 이미지 선택",
    fileHelp: "PNG, JPG, WEBP. 현재 파일:",
    runSample: "샘플 실행",
    pasteUrl: "URL 붙여넣기",
    chooseImageError: "이미지 파일을 선택하세요.",
    classifying: "분류 중",
    analysisResults: "분석 결과",
    matchesFound: "일치 후보",
    readyToAnalyze: "분석 준비 완료",
    matches: "개 후보",
    lowConfidence: "낮은 확신도",
    primaryMatch: "1순위 예측",
    topPrediction: "최상위 예측",
    confidence: "확신도",
    uncertainCopy: "모델의 확신도가 낮습니다. 복잡한 배경, 작은 피사체, 특이한 각도가 예측에 영향을 줄 수 있습니다.",
    strongCopy: "업로드한 이미지와 강한 시각적 일치 결과를 찾았습니다.",
    alternativeMatch: "대체 후보",
    recognitionTip: "인식 팁",
    tipBody: "정확도를 높이려면 포켓몬이 중앙에 있고 조명이 충분하며 배경이 복잡하지 않은 이미지를 사용하세요.",
    noAnalysis: "아직 분석 결과 없음",
    noAnalysisBody: "이미지를 선택하거나 샘플을 실행하면 예측 순위가 표시됩니다.",
  },
};

export default function App() {
  const [modelStatus, setModelStatus] = useState("loading");
  const [previewUrl, setPreviewUrl] = useState(SAMPLE_PATH);
  const [fileName, setFileName] = useState("test1.jpg");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isPredicting, setIsPredicting] = useState(false);
  const [thumbnails, setThumbnails] = useState({});
  const [language, setLanguage] = useState("en");
  const [pokemonKoNames, setPokemonKoNames] = useState({});

  const topPrediction = result?.predictions?.[0];
  const secondaryPredictions = result?.predictions?.slice(1, 4) ?? [];
  const isLowConfidence = topPrediction?.confidence < LOW_CONFIDENCE_THRESHOLD;
  const t = TEXT[language];

  const statusText = useMemo(() => {
    if (modelStatus === "ready") return t.modelReady;
    if (modelStatus === "error") return t.modelFailed;
    return t.modelLoading;
  }, [modelStatus, t]);

  useEffect(() => {
    loadModel()
      .then(() => setModelStatus("ready"))
      .catch((loadError) => {
        setModelStatus("error");
        setError(loadError.message);
      });
  }, []);

  useEffect(() => {
    fetch(POKEMON_KO_NAMES_URL)
      .then((response) => {
        if (!response.ok) return "{}";
        return response.text();
      })
      .then((text) => {
        try {
          setPokemonKoNames(JSON.parse(text));
        } catch {
          setPokemonKoNames({});
        }
      })
      .catch(() => setPokemonKoNames({}));
  }, []);

  useEffect(() => {
    fetch(THUMBNAIL_MANIFEST_URL)
      .then((response) => {
        if (!response.ok) return {};
        return response.json();
      })
      .then((manifest) => setThumbnails(manifest))
      .catch(() => setThumbnails({}));
  }, []);

  async function handleFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError(t.chooseImageError);
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

  return (
    <div className="app-page">
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="/">
            POKEDEX
          </a>
          <div className="topbar-actions">
            <div className={`model-status ${modelStatus}`}>
              <span />
              {statusText}
            </div>
            <div className="language-toggle" aria-label="Language selector">
              <button
                type="button"
                className={language === "ko" ? "active" : ""}
                onClick={() => setLanguage("ko")}
              >
                KO
              </button>
              <button
                type="button"
                className={language === "en" ? "active" : ""}
                onClick={() => setLanguage("en")}
              >
                EN
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="main-canvas">
        <section className="hero">
          <h1>{t.heroTitle}</h1>
          <p>{t.heroBody}</p>
        </section>

        <section className="scanner-card">
          <div className="scan-preview">
            <div className="preview-frame">
              <img src={previewUrl} alt="Selected Pokemon preview" />
              {isPredicting ? (
                <div className="scan-layer">
                  <span />
                  {t.classifying}
                </div>
              ) : null}
            </div>
          </div>

          <div className="upload-panel">
            <label className="dropzone">
              <input type="file" accept="image/*" onChange={handleFileChange} disabled={modelStatus !== "ready"} />
              <span className="camera-mark">+</span>
              <strong>{t.chooseImage}</strong>
              <small>{t.fileHelp} {fileName}</small>
            </label>

            <div className="button-row">
              <button type="button" onClick={handleSample} disabled={modelStatus !== "ready" || isPredicting}>
                {t.runSample}
              </button>
              <button type="button" className="secondary-button" disabled>
                {t.pasteUrl}
              </button>
            </div>

            {error ? <p className="error-box">{error}</p> : null}
          </div>
        </section>

        <section className="results-section">
          <div className="section-heading">
            <div>
              <p className="eyebrow">{t.analysisResults}</p>
              <h2>{topPrediction ? t.matchesFound : t.readyToAnalyze}</h2>
            </div>
            {topPrediction ? <span className="result-count">{result.predictions.length} {t.matches}</span> : null}
          </div>

          {topPrediction ? (
            <div className="result-grid">
              <article className={`primary-result ${isLowConfidence ? "low-confidence" : ""}`}>
                <div className="primary-image">
                  <img src={getPokemonImage(thumbnails, topPrediction.label, previewUrl)} alt={topPrediction.label} />
                  <span>{isLowConfidence ? t.lowConfidence : t.primaryMatch}</span>
                </div>
                <div className="primary-content">
                  <div className="result-title-row">
                    <div>
                      <p className="eyebrow">{t.topPrediction}</p>
                      <h3>{displayPokemonName(topPrediction.label, language, pokemonKoNames)}</h3>
                    </div>
                    <div className="confidence-score">
                      <strong>{formatPercent(topPrediction.confidence)}</strong>
                      <small>{t.confidence}</small>
                    </div>
                  </div>

                  <p className="result-copy">
                    {isLowConfidence ? t.uncertainCopy : t.strongCopy}
                  </p>

                  <div className="confidence-track">
                    <div style={{ width: formatPercent(topPrediction.confidence) }} />
                  </div>
                </div>
              </article>

              <aside className="side-results">
                {secondaryPredictions.map((prediction) => (
                  <article className="secondary-result" key={prediction.classIndex}>
                    <div>
                      <h4>{displayPokemonName(prediction.label, language, pokemonKoNames)}</h4>
                      <p>{t.alternativeMatch}</p>
                    </div>
                    <strong>{formatPercent(prediction.confidence)}</strong>
                    <div className="mini-track">
                      <div style={{ width: formatPercent(prediction.confidence) }} />
                    </div>
                  </article>
                ))}

                <article className="tip-card">
                  <p className="eyebrow">{t.recognitionTip}</p>
                  <p>{t.tipBody}</p>
                </article>
              </aside>
            </div>
          ) : (
            <div className="empty-result">
              <h3>{t.noAnalysis}</h3>
              <p>{t.noAnalysisBody}</p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function getPokemonImage(thumbnails, label, fallbackUrl) {
  return thumbnails[label] || fallbackUrl;
}

function displayPokemonName(label, language, pokemonKoNames) {
  if (language !== "ko") {
    return label;
  }

  return pokemonKoNames[label] || label;
}
