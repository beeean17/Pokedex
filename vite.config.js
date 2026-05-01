import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const rootDir = path.dirname(fileURLToPath(import.meta.url));
const onnxRuntimeDist = path.join(rootDir, "node_modules", "onnxruntime-web", "dist");

function onnxRuntimeWasm() {
  return {
    name: "onnxruntime-wasm",
    configureServer(server) {
      server.middlewares.use("/ort", (request, response, next) => {
        const url = new URL(request.url || "/", "http://localhost");
        const fileName = path.basename(decodeURIComponent(url.pathname));
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

export default defineConfig({
  plugins: [react(), onnxRuntimeWasm()],
});
