import fs from 'node:fs';
import path from 'node:path';
import type { Plugin } from 'vite';
import { defineConfig } from 'vite';

function serveStaticMappings(): Plugin {
  const spikeDir = __dirname;
  const repoRoot = path.resolve(spikeDir, '..');
  const modelsDir = path.join(repoRoot, 'web', 'public', 'models');
  const valImagesDir = path.join(repoRoot, 'data', 'yolo', 'grouped', 'val', 'images');
  const ortDistDir = path.join(spikeDir, 'node_modules', 'onnxruntime-web', 'dist');

  const mimeMap: Record<string, string> = {
    '.json': 'application/json; charset=utf-8',
    '.onnx': 'application/octet-stream',
    '.wasm': 'application/wasm',
    '.mjs': 'application/javascript; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
  };

  return {
    name: 'tracera-spike-static',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const urlPath = (req.url || '').split('?')[0];

        let targetDir: string | null = null;
        let relPath = '';
        if (urlPath.startsWith('/models/')) {
          targetDir = modelsDir;
          relPath = urlPath.slice('/models/'.length);
        } else if (urlPath.startsWith('/val-images/')) {
          targetDir = valImagesDir;
          relPath = urlPath.slice('/val-images/'.length);
        } else if (urlPath.startsWith('/ort/')) {
          targetDir = ortDistDir;
          relPath = urlPath.slice('/ort/'.length);
        } else if (urlPath === '/samples.json') {
          const sampleFile = path.join(spikeDir, 'samples.json');
          if (fs.existsSync(sampleFile)) {
            res.setHeader('Content-Type', 'application/json; charset=utf-8');
            fs.createReadStream(sampleFile).pipe(res);
            return;
          }
        }

        if (!targetDir) {
          next();
          return;
        }

        const safeRel = path.normalize(decodeURIComponent(relPath)).replace(/^(\.\.[/\\])+/, '');
        const fullPath = path.join(targetDir, safeRel);
        if (!fullPath.startsWith(targetDir) || !fs.existsSync(fullPath) || !fs.statSync(fullPath).isFile()) {
          res.statusCode = 404;
          res.end('Not found');
          return;
        }

        const ext = path.extname(fullPath).toLowerCase();
        res.setHeader('Content-Type', mimeMap[ext] || 'application/octet-stream');
        // Cross-origin isolation headers allow multi-threaded WASM if needed
        res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
        res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
        fs.createReadStream(fullPath).pipe(res);
      });
    },
  };
}

export default defineConfig({
  plugins: [serveStaticMappings()],
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    host: '127.0.0.1',
    port: 5179,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
