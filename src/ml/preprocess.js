export async function imageFileToTensor(file, labels) {
  const image = await loadImage(file);
  return imageToTensor(image, labels);
}

export function imageToTensor(image, labels) {
  const inputSize = Number(labels.inputSize);
  const canvas = document.createElement("canvas");
  canvas.width = inputSize;
  canvas.height = inputSize;

  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.fillStyle = "rgb(255, 255, 255)";
  context.fillRect(0, 0, inputSize, inputSize);

  const scale = Math.min(inputSize / image.naturalWidth, inputSize / image.naturalHeight);
  const width = Math.round(image.naturalWidth * scale);
  const height = Math.round(image.naturalHeight * scale);
  const left = Math.floor((inputSize - width) / 2);
  const top = Math.floor((inputSize - height) / 2);
  context.drawImage(image, left, top, width, height);

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
