import { imageFileToTensor } from "./preprocess.js";

const MODEL_URL = "/models/pokemon-efficientnet-b0-fp32.onnx";
const LABELS_URL = "/models/labels.v1.json";

let modelPromise;
let runtimePromise;

async function loadRuntime() {
  if (!runtimePromise) {
    runtimePromise = import("onnxruntime-web").then((runtime) => {
      runtime.env.wasm.numThreads = 1;
      runtime.env.wasm.wasmPaths = "/ort/";
      return runtime;
    });
  }

  return runtimePromise;
}

export async function loadModel() {
  if (!modelPromise) {
    modelPromise = Promise.all([
      fetch(LABELS_URL).then((response) => {
        if (!response.ok) {
          throw new Error("Could not load labels.");
        }
        return response.json();
      }),
      loadRuntime().then((ort) =>
        ort.InferenceSession.create(MODEL_URL, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        }),
      ),
    ]).then(([labels, session]) => ({ labels, session }));
  }

  return modelPromise;
}

export async function predictPokemon(file) {
  const [{ labels, session }, ort] = await Promise.all([loadModel(), loadRuntime()]);
  const startedAt = performance.now();
  const inputSize = Number(labels.inputSize);
  const tensorData = await imageFileToTensor(file, labels);
  const inputTensor = new ort.Tensor("float32", tensorData, [1, 3, inputSize, inputSize]);
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const outputs = await session.run({ [inputName]: inputTensor });
  const probabilities = softmax(outputs[outputName].data);

  return {
    elapsedMs: performance.now() - startedAt,
    predictions: topK(probabilities, labels.classes, 5),
  };
}

function softmax(logits) {
  let max = Number.NEGATIVE_INFINITY;
  for (const value of logits) {
    max = Math.max(max, value);
  }

  let sum = 0;
  const values = new Float32Array(logits.length);
  for (let index = 0; index < logits.length; index += 1) {
    const value = Math.exp(logits[index] - max);
    values[index] = value;
    sum += value;
  }

  return values.map((value) => value / sum);
}

function topK(probabilities, classes, k) {
  return Array.from(probabilities)
    .map((confidence, classIndex) => ({
      classIndex,
      label: classes[classIndex],
      confidence,
    }))
    .sort((left, right) => right.confidence - left.confidence)
    .slice(0, k)
    .map((prediction, index) => ({
      ...prediction,
      rank: index + 1,
    }));
}
