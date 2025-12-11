/// <reference lib="webworker" />

import {
  env,
  pipeline,
  type AutomaticSpeechRecognitionPipeline,
  type ChunkCallbackItem
} from '@xenova/transformers';

type WorkerConfig = {
  wasmPath: string;
  allowLocalModels: boolean;
  allowRemoteModels: boolean;
  useBrowserCache: boolean;
  localModelPath: string;
};

type TranscribePayload = WorkerConfig & {
  audio: ArrayBuffer;
  sampleRate: number;
  chunkLength: number;
  strideLength: number;
  returnTimestamps: boolean;
};

type TranscribeMessage = {
  type: 'transcribe';
  jobId: number;
  payload: TranscribePayload;
};

type WorkerResponse =
  | { type: 'progress'; jobId: number; processed: number; total: number }
  | { type: 'partial'; jobId: number; processed: number; total: number; chunks: { timestamp: [number, number]; text: string }[] }
  | { type: 'done'; jobId: number; chunks: { timestamp: [number, number]; text: string }[] }
  | { type: 'error'; jobId: number; message: string };

type DecodableChunk = Pick<ChunkCallbackItem, 'stride' | 'tokens' | 'token_timestamps' | 'is_last'>;

let cachedConfig: WorkerConfig | null = null;
let transcriberPromise: Promise<AutomaticSpeechRecognitionPipeline> | null = null;

const ctx: DedicatedWorkerGlobalScope = self as unknown as DedicatedWorkerGlobalScope;

const resampleToRate = (input: Float32Array, sourceRate: number, targetRate: number) => {
  if (sourceRate === targetRate) return input;
  const ratio = targetRate / sourceRate;
  const length = Math.max(1, Math.round(input.length * ratio));
  const output = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    const position = i / ratio;
    const index = Math.floor(position);
    const next = Math.min(index + 1, input.length - 1);
    const weight = position - index;
    output[i] = input[index] * (1 - weight) + input[next] * weight;
  }
  return output;
};

const estimateChunkTotal = (sampleCount: number, sampleRate: number, chunkLength: number, strideLength: number) => {
  if (chunkLength <= 0) return 1;
  const window = sampleRate * chunkLength;
  const stride = sampleRate * strideLength;
  const jump = window - 2 * stride;
  if (window <= 0) return 1;
  if (sampleCount <= window) return 1;
  if (jump <= 0) {
    return Math.ceil(sampleCount / window);
  }
  return Math.max(1, Math.ceil((sampleCount - window) / jump) + 1);
};

const ensureTranscriber = async (config: WorkerConfig) => {
  const needsRefresh =
    !cachedConfig ||
    cachedConfig.wasmPath !== config.wasmPath ||
    cachedConfig.allowLocalModels !== config.allowLocalModels ||
    cachedConfig.allowRemoteModels !== config.allowRemoteModels ||
    cachedConfig.useBrowserCache !== config.useBrowserCache ||
    cachedConfig.localModelPath !== config.localModelPath;

  if (!transcriberPromise || needsRefresh) {
    env.allowLocalModels = config.allowLocalModels;
    env.allowRemoteModels = config.allowRemoteModels;
    env.useBrowserCache = config.useBrowserCache;
    env.localModelPath = config.localModelPath;
    env.backends.onnx.wasm.wasmPaths = config.wasmPath;
    cachedConfig = { ...config };
    transcriberPromise = pipeline('automatic-speech-recognition', 'whisper-tiny') as Promise<
      AutomaticSpeechRecognitionPipeline
    >;
  }
  return transcriberPromise;
};

ctx.onmessage = async (event: MessageEvent<TranscribeMessage>) => {
  const { data } = event;
  if (data.type !== 'transcribe') return;

  const { payload, jobId } = data;

  try {
    const transcriber = await ensureTranscriber(payload);
    const expectedRate = transcriber.processor.feature_extractor.config.sampling_rate ?? payload.sampleRate;

    const sourceAudio = new Float32Array(payload.audio);
    const audio = payload.sampleRate === expectedRate ? sourceAudio : resampleToRate(sourceAudio, payload.sampleRate, expectedRate);

    const totalChunks = estimateChunkTotal(audio.length, expectedRate, payload.chunkLength, payload.strideLength);
    let processed = 0;
    const collected: DecodableChunk[] = [];

    const timePrecision =
      transcriber.processor.feature_extractor.config.chunk_length / transcriber.model.config.max_source_positions;

    const decodePartial = () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const [_, optional]: any = (transcriber as any).tokenizer._decode_asr(collected, {
        time_precision: timePrecision,
        return_timestamps: payload.returnTimestamps,
        force_full_sequences: false
      });
      const chunkList = (optional?.chunks ?? []) as { timestamp: [number, number]; text: string }[];
      return chunkList;
    };

    const result = await transcriber(audio, {
      chunk_length_s: payload.chunkLength,
      stride_length_s: payload.strideLength,
      return_timestamps: payload.returnTimestamps,
      chunk_callback: (chunk: ChunkCallbackItem) => {
        collected.push({
          stride: chunk.stride,
          tokens: chunk.tokens,
          token_timestamps: chunk.token_timestamps,
          is_last: chunk.is_last
        });
        processed += 1;
        const partialChunks = decodePartial();
        const message: WorkerResponse = {
          type: 'partial',
          jobId,
          processed,
          total: totalChunks,
          chunks: partialChunks
        };
        ctx.postMessage(message);
      }
    });

    const doneMessage: WorkerResponse = {
      type: 'done',
      jobId,
      chunks: (result.chunks as { timestamp: [number, number]; text: string }[]) ?? []
    };
    ctx.postMessage(doneMessage);
  } catch (error) {
    const message: WorkerResponse = {
      type: 'error',
      jobId,
      message: (error as Error)?.message ?? '转写失败'
    };
    ctx.postMessage(message);
  }
};
