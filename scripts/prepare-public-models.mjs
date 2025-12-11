import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const ortDst = path.join(projectRoot, 'public', 'models', 'onnx-runtime');

async function findOrtDist() {
  const pnpmDir = path.join(projectRoot, 'node_modules', '.pnpm');
  const entries = await fs.readdir(pnpmDir, { withFileTypes: true }).catch(() => []);
  const ortEntry = entries.find((e) => e.isDirectory() && e.name.startsWith('onnxruntime-web@'));
  if (!ortEntry) {
    throw new Error('未找到 onnxruntime-web 安装目录，请先执行 pnpm install');
  }
  return path.join(pnpmDir, ortEntry.name, 'node_modules', 'onnxruntime-web', 'dist');
}

const ortSrc = await findOrtDist();

const keepOrtFiles = [
  'ort-wasm-simd.wasm',
  'ort-wasm.wasm',
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-threaded.wasm',
  'ort-wasm-threaded.js',
  'ort-wasm-threaded.worker.js',
  'ort-web.es6.min.js',
  'ort-web.es5.min.js'
];

async function ensureDir(dir) {
  await fs.mkdir(dir, { recursive: true });
}

async function cleanDirExcept(dir, keep) {
  try {
    const entries = await fs.readdir(dir);
    const keepSet = new Set(keep);
    for (const entry of entries) {
      if (!keepSet.has(entry)) {
        await fs.rm(path.join(dir, entry), { recursive: true, force: true });
      }
    }
  } catch {
    // ignore if dir does not exist
  }
}

async function copyList(srcDir, dstDir, files) {
  await ensureDir(dstDir);
  for (const file of files) {
    const from = path.join(srcDir, file);
    const to = path.join(dstDir, file);
    await fs.copyFile(from, to);
  }
}

async function main() {
  await ensureDir(ortDst);
  await cleanDirExcept(ortDst, keepOrtFiles);
  await copyList(ortSrc, ortDst, keepOrtFiles);
  console.log('Prepared minimal ONNX Runtime files in public/models/onnx-runtime');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
