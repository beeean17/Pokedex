import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const rootDir = path.dirname(fileURLToPath(import.meta.url));
const onnxRuntimeDist = path.join(rootDir, "node_modules", "onnxruntime-web", "dist");
const pokemonDataDir = path.join(rootDir, "PokemonData");
const thumbnailRoute = "/pokemon-thumbnails";
const imageExtensions = new Set([".jpg", ".jpeg", ".png", ".webp"]);

function onnxRuntimeWasm() {
  return {
    name: "onnxruntime-wasm",
    configureServer(server) {
      server.middlewares.use("/ort", (request, response, next) => {
        const url = new URL(request.url || "/", "http://localhost");
        const fileName = path.basename(url.pathname);
        const isWasm = fileName.endsWith(".wasm");
        const isModule = fileName.endsWith(".mjs");

        if (!isWasm && !isModule) {
          next();
          return;
        }

        const wasmPath = path.join(onnxRuntimeDist, path.basename(fileName));
        if (!fs.existsSync(wasmPath)) {
          next();
          return;
        }

        response.setHeader("Content-Type", isWasm ? "application/wasm" : "text/javascript");
        fs.createReadStream(wasmPath).pipe(response);
      });
    },
    closeBundle() {
      const outputDir = path.join(rootDir, "dist", "ort");
      if (!fs.existsSync(onnxRuntimeDist)) {
        return;
      }

      fs.mkdirSync(outputDir, { recursive: true });
      for (const fileName of fs.readdirSync(onnxRuntimeDist)) {
        if (fileName.endsWith(".wasm") || fileName.endsWith(".mjs")) {
          fs.copyFileSync(path.join(onnxRuntimeDist, fileName), path.join(outputDir, fileName));
        }
      }
    },
  };
}

function contentTypeFor(filePath) {
  const extension = path.extname(filePath).toLowerCase();
  if (extension === ".png") return "image/png";
  if (extension === ".webp") return "image/webp";
  return "image/jpeg";
}

function readPokemonThumbnails() {
  if (!fs.existsSync(pokemonDataDir)) {
    return { manifest: {}, files: new Map() };
  }

  const manifest = {};
  const files = new Map();
  const classDirs = fs
    .readdirSync(pokemonDataDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .sort((left, right) => left.name.localeCompare(right.name));

  for (const classDir of classDirs) {
    const classPath = path.join(pokemonDataDir, classDir.name);
    const firstImage = fs
      .readdirSync(classPath, { withFileTypes: true })
      .filter((entry) => entry.isFile() && imageExtensions.has(path.extname(entry.name).toLowerCase()))
      .sort((left, right) => left.name.localeCompare(right.name))[0];

    if (!firstImage) {
      continue;
    }

    const extension = path.extname(firstImage.name).toLowerCase();
    const outputName = `${encodeURIComponent(classDir.name)}${extension}`;
    const url = `${thumbnailRoute}/${outputName}`;
    const filePath = path.join(classPath, firstImage.name);

    manifest[classDir.name] = url;
    files.set(outputName, filePath);
  }

  return { manifest, files };
}

function pokemonThumbnails() {
  return {
    name: "pokemon-thumbnails",
    configureServer(server) {
      server.middlewares.use(thumbnailRoute, (request, response, next) => {
        const url = new URL(request.url || "/", "http://localhost");
        const fileName = path.basename(decodeURIComponent(url.pathname));
        const { manifest, files } = readPokemonThumbnails();

        if (fileName === "manifest.json") {
          response.setHeader("Content-Type", "application/json");
          response.end(JSON.stringify(manifest));
          return;
        }

        const imagePath = files.get(fileName);
        if (!imagePath) {
          next();
          return;
        }

        response.setHeader("Content-Type", contentTypeFor(imagePath));
        fs.createReadStream(imagePath).pipe(response);
      });
    },
    closeBundle() {
      const { manifest, files } = readPokemonThumbnails();
      const outputDir = path.join(rootDir, "dist", thumbnailRoute.slice(1));
      fs.mkdirSync(outputDir, { recursive: true });
      fs.writeFileSync(path.join(outputDir, "manifest.json"), JSON.stringify(manifest, null, 2));

      for (const [fileName, imagePath] of files.entries()) {
        fs.copyFileSync(imagePath, path.join(outputDir, fileName));
      }
    },
  };
}

export default defineConfig({
  base: process.env.VITE_BASE_PATH || "/",
  plugins: [react(), onnxRuntimeWasm(), pokemonThumbnails()],
});
