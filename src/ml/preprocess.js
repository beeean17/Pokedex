export async function imageFileToTensor(file, labels) {
  const image = await loadImage(file);
  return imageToTensor(image, labels);
}

export async function imageFileToTensors(file, labels) {
  const image = await loadImage(file);
  return imageToTensors(image, labels);
}

export function imageToTensors(image, labels) {
  const views = [
    { name: "full", kind: "letterbox" },
    { name: "object", kind: "crop", scale: 0.86, centerX: 0.5, centerY: 0.5 },
    { name: "feature", kind: "crop", scale: 0.58, centerX: 0.5, centerY: 0.42 },
  ];
  const tensors = views.map((view) => imageToTensor(image, labels, view));
  const tensorLength = tensors[0].length;
  const data = new Float32Array(tensorLength * tensors.length);

  tensors.forEach((tensor, index) => {
    data.set(tensor, index * tensorLength);
  });

  return {
    data,
    views: views.map((view) => view.name),
  };
}

export function imageToTensor(image, labels, view = { kind: "letterbox" }) {
  const inputSize = Number(labels.inputSize);
  const canvas = document.createElement("canvas");
  canvas.width = inputSize;
  canvas.height = inputSize;

  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.fillStyle = "rgb(255, 255, 255)";
  context.fillRect(0, 0, inputSize, inputSize);

  if (view.kind === "crop") {
    drawCrop(context, image, inputSize, view);
  } else {
    drawLetterbox(context, image, inputSize);
  }

  const pixels = context.getImageData(0, 0, inputSize, inputSize).data;
  const channels = 3;
  const tensor = new Float32Array(channels * inputSize * inputSize);
  const mean = labels.mean;
  const std = labels.std;

  for (let y = 0; y < inputSize; y += 1) {
    for (let x = 0; x < inputSize; x += 1) {
      const pixelIndex = (y * inputSize + x) * 4;
      const tensorIndex = y * inputSize + x;
      tensor[tensorIndex] = pixels[pixelIndex] / 255 / std[0] - mean[0] / std[0];
      tensor[inputSize * inputSize + tensorIndex] = pixels[pixelIndex + 1] / 255 / std[1] - mean[1] / std[1];
      tensor[2 * inputSize * inputSize + tensorIndex] = pixels[pixelIndex + 2] / 255 / std[2] - mean[2] / std[2];
    }
  }

  return tensor;
}

function drawLetterbox(context, image, inputSize) {
  const imageWidth = getImageWidth(image);
  const imageHeight = getImageHeight(image);
  const scale = Math.min(inputSize / imageWidth, inputSize / imageHeight);
  const width = Math.round(imageWidth * scale);
  const height = Math.round(imageHeight * scale);
  const left = Math.floor((inputSize - width) / 2);
  const top = Math.floor((inputSize - height) / 2);
  context.drawImage(image, left, top, width, height);
}

function drawCrop(context, image, inputSize, view) {
  const imageWidth = getImageWidth(image);
  const imageHeight = getImageHeight(image);
  const side = Math.max(1, Math.round(Math.min(imageWidth, imageHeight) * view.scale));
  const left = clamp(Math.round(imageWidth * view.centerX - side / 2), 0, imageWidth - side);
  const top = clamp(Math.round(imageHeight * view.centerY - side / 2), 0, imageHeight - side);
  context.drawImage(image, left, top, side, side, 0, 0, inputSize, inputSize);
}

function getImageWidth(image) {
  return image.naturalWidth || image.width;
}

function getImageHeight(image) {
  return image.naturalHeight || image.height;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("이미지를 불러오지 못했습니다."));
    };
    image.src = url;
  });
}
