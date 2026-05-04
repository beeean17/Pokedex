import { imageFileToTensors } from "./preprocess.js";

const BASE_URL = import.meta.env.BASE_URL;
const MODEL_URL = `${BASE_URL}models/pokemon-efficientnet-b0-fp32.onnx`;
const LABELS_URL = `${BASE_URL}models/labels.v1.json`;

let modelPromise;
let runtimePromise;

async function loadRuntime() {
  if (!runtimePromise) {
    runtimePromise = import("onnxruntime-web").then((runtime) => {
      runtime.env.wasm.numThreads = 1;
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
  const { data, views } = await imageFileToTensors(file, labels);
  const inputTensor = new ort.Tensor("float32", data, [views.length, 3, inputSize, inputSize]);
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const outputs = await session.run({ [inputName]: inputTensor });
  const logits = averageLogits(outputs[outputName].data, views.length);
  const probabilities = softmax(logits);

  return {
    elapsedMs: performance.now() - startedAt,
    views,
    predictions: topK(probabilities, labels.classes, 5),
  };
}

function averageLogits(logits, batchSize) {
  const classCount = Math.floor(logits.length / batchSize);
  const averaged = new Float32Array(classCount);

  for (let batchIndex = 0; batchIndex < batchSize; batchIndex += 1) {
    const offset = batchIndex * classCount;
    for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
      averaged[classIndex] += logits[offset + classIndex] / batchSize;
    }
  }

  return averaged;
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
