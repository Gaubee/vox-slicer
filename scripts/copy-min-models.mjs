import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');
const distRoot = path.join(projectRoot, 'dist');

const whisperSrc = path.join(projectRoot, 'public', 'models', 'whisper-tiny');
// 使用 Xenova/whisper-tiny 路径以匹配 @xenova/transformers 的模型查找路径
const whisperDst = path.join(distRoot, 'models', 'Xenova', 'whisper-tiny');
const ortSrc = path.join(projectRoot, 'public', 'models', 'onnx-runtime');
const ortDst = path.join(distRoot, 'models', 'onnx-runtime');

const whisperFiles = [
  'config.json',
  'generation_config.json',
  'preprocessor_config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'vocab.json',
  'merges.txt',
  'special_tokens_map.json',
  'added_tokens.json',
  'normalizer.json',
  'quantize_config.json'
];

const whisperOnnxFiles = [
  'encoder_model_quantized.onnx',
  'decoder_model_merged_quantized.onnx'
];

const ortFiles = [
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

async function copyList(srcDir, dstDir, files) {
  await ensureDir(dstDir);
  for (const file of files) {
    const from = path.join(srcDir, file);
    const to = path.join(dstDir, file);
    await fs.copyFile(from, to);
  }
}

async function main() {
  await ensureDir(distRoot);
  await copyList(whisperSrc, whisperDst, whisperFiles);
  await copyList(path.join(whisperSrc, 'onnx'), path.join(whisperDst, 'onnx'), whisperOnnxFiles);
  await copyList(ortSrc, ortDst, ortFiles);
  console.log('Copied minimal model/runtime files into dist/models.');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
