import React, { useEffect, useMemo, useRef, useState } from "react";
import { loadModel, predictPokemon } from "./ml/predict.js";

const BASE_URL = import.meta.env.BASE_URL;
const SAMPLE_PATH = `${BASE_URL}sample/test1.jpg`;
const LOW_CONFIDENCE_THRESHOLD = 0.45;
const POKEMON_KO_NAMES_URL = `${BASE_URL}i18n/pokemon-ko.json`;

const TEXT = {
  en: {
    heroBody:
      "Upload a Pokemon image and compare the top five local ONNX predictions in one focused workspace.",
    modelReady: "Model ready",
    modelFailed: "Model failed",
    modelLoading: "Loading model",
    chooseImage: "Choose image",
    fileHelp: "PNG, JPG, WEBP",
    dropHelp: "PNG, JPG, WEBP · drag and drop supported",
    currentFile: "Current file",
    chooseImageError: "Choose an image file.",
    classifying: "Classifying",
    uploadTitle: "Uploaded image",
    uploadBody: "Keep the source image visible while checking each ranked match.",
    analysisResults: "Analysis results",
    matchesFound: "Top 5 candidates",
    readyToAnalyze: "Ready to analyze",
    matches: "matches",
    lowConfidence: "Low confidence",
    primaryMatch: "Most likely",
    topPrediction: "Rank 1",
    confidence: "confidence",
    uncertainCopy:
      "The model is uncertain. A busy background, small subject, or unusual angle may be affecting the prediction.",
    strongCopy: "The model found a strong visual match for the uploaded specimen.",
    alternativeMatch: "Candidate",
    noAnalysis: "No analysis yet",
    noAnalysisBody: "Choose an image or run the sample to see five ranked predictions here.",
    uploadedImage: "Uploaded image",
    closePreview: "Close preview",
  },
  ko: {
    heroBody:
      "포켓몬 이미지를 올리면 브라우저 안에서 ONNX 추론을 실행하고, 상위 5개 후보를 한 화면에서 비교합니다.",
    modelReady: "모델 준비 완료",
    modelFailed: "모델 로드 실패",
    modelLoading: "모델 로딩 중",
    chooseImage: "이미지 선택",
    fileHelp: "PNG, JPG, WEBP",
    dropHelp: "PNG, JPG, WEBP · 드래그 앤 드롭 지원",
    currentFile: "현재 파일",
    chooseImageError: "이미지 파일을 선택하세요.",
    classifying: "분류 중",
    uploadTitle: "업로드 이미지",
    uploadBody: "원본 이미지를 고정해 두고 순위별 후보를 바로 비교하세요.",
    analysisResults: "분석 결과",
    matchesFound: "상위 5개 후보",
    readyToAnalyze: "분석 준비 완료",
    matches: "개 후보",
    lowConfidence: "낮은 확신도",
    primaryMatch: "가장 유력",
    topPrediction: "1위 후보",
    confidence: "일치율",
    uncertainCopy:
      "모델의 확신도가 낮습니다. 복잡한 배경, 작은 피사체, 특이한 각도가 예측에 영향을 줬을 수 있습니다.",
    strongCopy: "업로드 이미지와 강하게 닮은 시각적 후보를 찾았습니다.",
    alternativeMatch: "후보",
    noAnalysis: "아직 분석 결과가 없습니다",
    noAnalysisBody: "이미지를 선택하거나 샘플을 실행하면 순위별 예측이 표시됩니다.",
    uploadedImage: "업로드 이미지",
    closePreview: "미리보기 닫기",
  },
};

export default function App() {
  const [modelStatus, setModelStatus] = useState("loading");
  const [previewUrl, setPreviewUrl] = useState(SAMPLE_PATH);
  const [fileName, setFileName] = useState("test1.jpg");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isPredicting, setIsPredicting] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [language, setLanguage] = useState("ko");
  const [pokemonKoNames, setPokemonKoNames] = useState({});
  const [previewDialog, setPreviewDialog] = useState(null);
  const resultsRef = useRef(null);
  const initialSampleStarted = useRef(false);

  const topPrediction = result?.predictions?.[0];
  const topFivePredictions = result?.predictions?.slice(0, 5) ?? [];
  const secondaryPredictions = topFivePredictions.slice(1, 5);
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
    if (!topPrediction || isPredicting) return;
    resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [topPrediction, isPredicting]);

  useEffect(() => {
    if (modelStatus !== "ready" || initialSampleStarted.current) return;
    initialSampleStarted.current = true;
    runSamplePrediction();
  }, [modelStatus]);

  useEffect(() => {
    return () => {
      if (previewUrl?.startsWith("blob:")) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  async function handleFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    await processImageFile(file);
    event.target.value = "";
  }

  async function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    if (modelStatus !== "ready") return;

    const file = event.dataTransfer.files?.[0];
    if (!file) return;
    await processImageFile(file);
  }

  function handleDragOver(event) {
    event.preventDefault();
    if (modelStatus === "ready") {
      setIsDragging(true);
    }
  }

  function handleDragLeave(event) {
    if (!event.currentTarget.contains(event.relatedTarget)) {
      setIsDragging(false);
    }
  }

  async function processImageFile(file) {
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

  async function runSamplePrediction() {
    try {
      setResult(null);
      setError("");
      setFileName("test1.jpg");
      setPreviewUrl((previousUrl) => {
        if (previousUrl?.startsWith("blob:")) URL.revokeObjectURL(previousUrl);
        return SAMPLE_PATH;
      });

      const response = await fetch(SAMPLE_PATH);
      if (!response.ok) {
        throw new Error("Could not load the sample image.");
      }

      const blob = await response.blob();
      const file = new File([blob], "test1.jpg", { type: blob.type || "image/jpeg" });
      await runPrediction(file, true);
    } catch (sampleError) {
      setError(sampleError.message);
    }
  }

  async function runPrediction(file, force = false) {
    if (!force && modelStatus !== "ready") return;

    setIsPredicting(true);
    try {
      setResult(await predictPokemon(file));
    } catch (predictionError) {
      setError(predictionError.message);
    } finally {
      setIsPredicting(false);
    }
  }

  function openPreview(src, title) {
    setPreviewDialog({ src, title });
  }

  return (
    <div className="app-page">
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="/">
            <span className="brand-mark" aria-hidden="true" />
            PokeFInder
          </a>
          <div className="topbar-actions">
            <div className={`model-status ${modelStatus}`} aria-label={statusText}>
              <span className="status-light" />
              <span className="status-label">{statusText}</span>
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
        

        <section className="finder-dashboard" aria-label="Pokemon image classifier">
          <aside className="input-column">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">{t.uploadTitle}</p>
                <h2>{fileName}</h2>
              </div>
            </div>

            <button
              type="button"
              className="preview-button"
              onClick={() => openPreview(previewUrl, t.uploadedImage)}
              aria-label={t.uploadedImage}
            >
              <img src={previewUrl} alt="Selected Pokemon preview" />
              {isPredicting ? (
                <span className="scan-layer">
                  <span />
                  {t.classifying}
                </span>
              ) : null}
            </button>

            
            <label
              className={`dropzone ${isDragging ? "dragging" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <input type="file" accept="image/*" onChange={handleFileChange} disabled={modelStatus !== "ready"} />
              <span className="camera-mark">+</span>
              <span>
                <strong>{t.chooseImage}</strong>
                <small>{t.dropHelp || t.fileHelp}</small>
              </span>
            </label>

            <p className="file-meta">
              {t.currentFile}: <span>{fileName}</span>
            </p>

            {error ? <p className="error-box">{error}</p> : null}
          </aside>

          <section className="results-panel" ref={resultsRef}>
            <div className="panel-heading">
              <div>
                <p className="eyebrow">{t.analysisResults}</p>
                <h2>{topPrediction ? t.matchesFound : t.readyToAnalyze}</h2>
              </div>
              {topPrediction ? <span className="result-count">{topFivePredictions.length} {t.matches}</span> : null}
            </div>

            {topPrediction ? (
              <div className="ranked-results">
                <PrimaryResult
                  prediction={topPrediction}
                  language={language}
                  pokemonKoNames={pokemonKoNames}
                  isLowConfidence={isLowConfidence}
                  text={t}
                />

                <div className="secondary-grid">
                  {secondaryPredictions.map((prediction, index) => (
                    <SecondaryResult
                      key={prediction.classIndex}
                      rank={index + 2}
                      prediction={prediction}
                      language={language}
                      pokemonKoNames={pokemonKoNames}
                      text={t}
                    />
                  ))}
                </div>
              </div>
            ) : (
              <div className="empty-result">
                <h3>{t.noAnalysis}</h3>
                <p>{t.noAnalysisBody}</p>
              </div>
            )}
          </section>
        </section>
      </main>

      {previewDialog ? (
        <div className="dialog-backdrop" role="presentation" onClick={() => setPreviewDialog(null)}>
          <div
            className="image-dialog"
            role="dialog"
            aria-modal="true"
            aria-label={previewDialog.title}
            onClick={(event) => event.stopPropagation()}
          >
            <button type="button" className="dialog-close" onClick={() => setPreviewDialog(null)}>
              {t.closePreview}
            </button>
            <img src={previewDialog.src} alt={previewDialog.title} />
            <h2>{previewDialog.title}</h2>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function PrimaryResult({
  prediction,
  language,
  pokemonKoNames,
  isLowConfidence,
  text,
}) {
  const name = displayPokemonName(prediction.label, language, pokemonKoNames);
  const tone = confidenceTone(prediction.confidence);
  const accent = accentForLabel(prediction.label);

  return (
    <article className={`primary-result ${isLowConfidence ? "low-confidence" : ""}`} style={accent}>
      <div className="primary-content">
        <div className="result-title-row">
          <div>
            <p className="eyebrow">{text.topPrediction}</p>
            <h3>{name}</h3>
          </div>
          <CircularProgress percent={prediction.confidence} tone={tone} label={text.confidence} />
        </div>

        <span className="rank-badge">{isLowConfidence ? text.lowConfidence : text.primaryMatch}</span>

        <p className="result-copy">
          {isLowConfidence ? text.uncertainCopy : text.strongCopy}
        </p>

        <ConfidenceBar percent={prediction.confidence} tone={tone} />
      </div>
    </article>
  );
}

function SecondaryResult({
  rank,
  prediction,
  language,
  pokemonKoNames,
  text,
}) {
  const name = displayPokemonName(prediction.label, language, pokemonKoNames);
  const tone = confidenceTone(prediction.confidence);
  const accent = accentForLabel(prediction.label);

  return (
    <article className={`secondary-result ${tone}`} style={accent}>
      <div className="secondary-body">
        <div className="secondary-title">
          <span>#{rank}</span>
          <strong>{formatPercent(prediction.confidence)}</strong>
        </div>
        <h4>{name}</h4>
        <p>{text.alternativeMatch}</p>
        <ConfidenceBar percent={prediction.confidence} tone={tone} small />
      </div>
    </article>
  );
}

function CircularProgress({ percent, tone, label }) {
  const degrees = Math.round(percent * 360);

  return (
    <div className={`confidence-ring ${tone}`} style={{ "--progress": `${degrees}deg` }}>
      <strong>{formatPercent(percent)}</strong>
      <small>{label}</small>
    </div>
  );
}

function ConfidenceBar({ percent, tone, small = false }) {
  return (
    <div className={`confidence-track ${tone} ${small ? "small" : ""}`}>
      <div style={{ width: formatPercent(percent) }} />
    </div>
  );
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function displayPokemonName(label, language, pokemonKoNames) {
  if (language !== "ko") {
    return label;
  }

  return pokemonKoNames[label] || label;
}

function confidenceTone(value) {
  if (value >= 0.8) return "very-high";
  if (value >= 0.55) return "high";
  if (value >= 0.35) return "medium";
  return "low";
}

function accentForLabel(label) {
  const hue = [...label].reduce((total, character) => total + character.charCodeAt(0), 0) % 360;

  return {
    "--pokemon-accent": `hsl(${hue} 72% 44%)`,
    "--pokemon-soft": `hsl(${hue} 82% 94%)`,
    "--pokemon-wash": `hsl(${hue} 78% 97%)`,
  };
}
