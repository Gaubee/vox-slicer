<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue';
import JSZip from 'jszip';
import * as lamejs from 'lamejs';
import {
  AudioLines,
  Sparkles,
  Download,
  UploadCloud,
  PlayCircle,
  Link2,
  Link2Off,
  Timer,
  ListOrdered,
  PieChart,
  Info,
  Loader2
} from 'lucide-vue-next';
// inline worker 确保单文件输出时不额外生成 chunk
// eslint-disable-next-line import/no-unresolved
import TranscribeWorker from './workers/transcribe.worker?worker&inline';

type WorkerChunk = { timestamp: [number, number]; text: string };
type WorkerMessage =
  | { type: 'progress'; jobId: number; processed: number; total: number }
  | { type: 'partial'; jobId: number; processed: number; total: number; chunks: WorkerChunk[] }
  | { type: 'done'; jobId: number; chunks: WorkerChunk[] }
  | { type: 'error'; jobId: number; message: string };

type Segment = {
  id: number;
  start: number;
  end: number;
  text: string;
  selected: boolean;
  linkNext: boolean;
};

type AutoGroupMode = 'time' | 'count' | 'total';

const hostname = window.location.hostname;
const isLocalhost = ['localhost', '127.0.0.1', '::1'].includes(hostname);

const wasmBases = isLocalhost
  ? [
      'models/onnx-runtime/', // 本地优先
      'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/',
      'https://unpkg.com/onnxruntime-web/dist/',
      'https://cdn.bootcdn.net/ajax/libs/onnxruntime-web/1.14.0/'
    ]
  : [
      'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/',
      'https://unpkg.com/onnxruntime-web/dist/',
      'https://cdn.bootcdn.net/ajax/libs/onnxruntime-web/1.14.0/', // 国内 CDN 备用
      'models/onnx-runtime/' // 自托管兜底
    ];

const pickWasmPath = async () => {
  for (const base of wasmBases) {
    if (base.startsWith('models/')) return base; // 本地路径，无需探测
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 1500);
      const probeUrl = new URL('ort-wasm-simd.wasm', base).toString();
      const res = await fetch(probeUrl, { method: 'HEAD', signal: controller.signal });
      clearTimeout(timer);
      if (res.ok) return base;
    } catch (_error) {
      // 忽略，继续尝试下一条 CDN
    }
  }
  return 'models/onnx-runtime/';
};

const wasmPathPromise = pickWasmPath();

const audioBuffer = ref<AudioBuffer | null>(null);
const segments = ref<Segment[]>([]);
const status = ref('');
const isLoading = ref(false);
const isExporting = ref(false);
const selectAll = ref(true);

const showAutoGroupModal = ref(false);
const autoGroupMode = ref<AutoGroupMode>('time');
const autoGroupValue = ref(10);

const TARGET_SAMPLE_RATE = 16000;
const CHUNK_LENGTH_S = 15; // 更短的窗口便于细粒度进度与部分输出
const STRIDE_LENGTH_S = 5;

const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
let currentSource: AudioBufferSourceNode | null = null;
let transcribeWorker: Worker | null = null;
let activeJobId = 0;

const processedChunks = ref(0);
const totalChunks = ref(0);
const isTranscribing = ref(false);
const progressPercent = computed(() => {
  if (!totalChunks.value) return 0;
  return Math.min(100, Math.round((processedChunks.value / totalChunks.value) * 100));
});

const colorPalette = [
  { bg: 'bg-white', accent: 'bg-gray-200' },
  { bg: 'bg-blue-50', accent: 'bg-blue-300' },
  { bg: 'bg-green-50', accent: 'bg-green-300' },
  { bg: 'bg-amber-50', accent: 'bg-amber-300' },
  { bg: 'bg-purple-50', accent: 'bg-purple-300' },
  { bg: 'bg-pink-50', accent: 'bg-pink-300' }
];

const segmentsWithColors = computed(() => {
  let groupIndex = 0;
  return segments.value.map((seg) => {
    const color = colorPalette[groupIndex % colorPalette.length];
    const item = {
      ref: seg,
      bgColorClass: color.bg,
      accentColorClass: color.accent,
      isEndOfGroup: !seg.linkNext
    };
    if (!seg.linkNext) groupIndex += 1;
    return item;
  });
});

const groupCount = computed(() => segments.value.filter((s) => !s.linkNext).length || (segments.value.length ? 1 : 0));

const inputLabel = computed(() => {
  if (autoGroupMode.value === 'time') return '目标每组时长';
  if (autoGroupMode.value === 'count') return '每组包含句数';
  return '期望总组数';
});

const inputUnit = computed(() => {
  if (autoGroupMode.value === 'time') return '秒 (s)';
  if (autoGroupMode.value === 'count') return '句';
  return '组';
});

const inputHint = computed(() => {
  if (autoGroupMode.value === 'time') return '按时长连续合并句子，超过设定时长即换组。';
  if (autoGroupMode.value === 'count') return '固定每组包含的句子数量，多余的自动换组。';
  return '将所有句子均分为指定组数，尽量保持组大小接近。';
});

const formatTime = (seconds: number) => {
  const mins = Math.floor(seconds / 60)
    .toString()
    .padStart(2, '0');
  const secs = (seconds % 60).toFixed(2).padStart(5, '0');
  return `${mins}:${secs}`;
};

const syncSegmentsFromChunks = (chunks: WorkerChunk[]) => {
  const duration = audioBuffer.value?.duration ?? 0;
  segments.value = chunks.map((chunk, idx) => ({
    id: idx + 1,
    start: chunk.timestamp[0],
    end: chunk.timestamp[1] ?? duration,
    text: chunk.text.trim(),
    selected: true,
    linkNext: false
  }));
  selectAll.value = true;
};

const mixToMono = (buffer: AudioBuffer) => {
  if (buffer.numberOfChannels === 1) return buffer.getChannelData(0);
  const left = buffer.getChannelData(0);
  const right = buffer.getChannelData(1);
  const length = Math.min(left.length, right.length);
  const mono = new Float32Array(length);
  const scaling = Math.sqrt(2);
  for (let i = 0; i < length; i += 1) {
    mono[i] = (left[i] + right[i]) / scaling;
  }
  return mono;
};

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

const attachWorker = async () => {
  if (!transcribeWorker) {
    transcribeWorker = new TranscribeWorker();
    transcribeWorker.onmessage = (event: MessageEvent<WorkerMessage>) => {
      const data = event.data;
      if ('jobId' in data && data.jobId !== activeJobId) return;

      if (data.type === 'progress') {
        processedChunks.value = data.processed;
        totalChunks.value = data.total;
        status.value = `正在分析语音内容... (${data.processed}/${data.total})`;
        return;
      }

      if (data.type === 'partial') {
        processedChunks.value = data.processed;
        totalChunks.value = data.total;
        syncSegmentsFromChunks(data.chunks);
        status.value = `正在分析语音内容... (${data.processed}/${data.total})`;
        return;
      }

      if (data.type === 'done') {
        syncSegmentsFromChunks(data.chunks);
        status.value = '';
        isLoading.value = false;
        isTranscribing.value = false;
        processedChunks.value = data.chunks.length;
        totalChunks.value = data.chunks.length;
        return;
      }

      if (data.type === 'error') {
        status.value = data.message || '处理失败，请稍后重试。';
        isLoading.value = false;
        isTranscribing.value = false;
        return;
      }
    };
  }
  return transcribeWorker;
};

const stopPlayback = () => {
  try {
    currentSource?.stop();
  } catch (error) {
    console.warn('停止播放时出错', error);
  }
  currentSource = null;
};

onBeforeUnmount(stopPlayback);

const handleFileUpload = async (event: Event) => {
  const target = event.target as HTMLInputElement | null;
  const file = target?.files?.[0];
  if (!file) return;

  status.value = '正在解码音频...';
  isLoading.value = true;
  isTranscribing.value = false;
  segments.value = [];
  processedChunks.value = 0;
  totalChunks.value = 0;
  stopPlayback();

  try {
    const arrayBuffer = await file.arrayBuffer();
    audioBuffer.value = await audioContext.decodeAudioData(arrayBuffer.slice(0));

    const mono = mixToMono(audioBuffer.value);
    const resampled = resampleToRate(mono, audioBuffer.value.sampleRate, TARGET_SAMPLE_RATE);
    const worker = await attachWorker();
    if (!worker) {
      throw new Error('无法初始化转写 Worker');
    }

    activeJobId += 1;
    const jobId = activeJobId;
    processedChunks.value = 0;
    totalChunks.value = 0;
    isTranscribing.value = true;
    status.value = '正在分析语音内容...';

    const wasmPath = await wasmPathPromise;
    const baseUrl = window.location.href;
    const resolvedWasmPath = wasmPath.startsWith('http')
      ? wasmPath
      : new URL(wasmPath, baseUrl).toString();
    const resolvedLocalModelPath = new URL('models/', baseUrl).toString();

    worker.postMessage(
      {
        type: 'transcribe',
        jobId,
        payload: {
          audio: resampled.buffer,
          sampleRate: TARGET_SAMPLE_RATE,
          chunkLength: CHUNK_LENGTH_S,
          strideLength: STRIDE_LENGTH_S,
          returnTimestamps: true,
          wasmPath: resolvedWasmPath,
          localModelPath: resolvedLocalModelPath,
          useBrowserCache: true,
          isLocalhost
        }
      },
      [resampled.buffer]
    );
  } catch (error) {
    console.error(error);
    status.value = '处理失败，请确认模型文件已就绪且浏览器支持 WebAudio。';
    isTranscribing.value = false;
    isLoading.value = false;
  } finally {
    if (!isTranscribing.value) {
      isLoading.value = false;
    }
  }
};

const playSegment = (seg: Segment) => {
  if (!audioBuffer.value) return;
  stopPlayback();
  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer.value;
  source.connect(audioContext.destination);
  const duration = Math.max(0.01, seg.end - seg.start);
  source.start(0, seg.start, duration);
  currentSource = source;
};

const toggleLink = (index: number) => {
  const target = segments.value[index];
  if (target) target.linkNext = !target.linkNext;
};

const toggleSelectAll = () => {
  segments.value.forEach((s) => {
    s.selected = selectAll.value;
  });
};

const resetGroupingState = () => {
  segments.value.forEach((s) => {
    s.linkNext = false;
    s.selected = true;
  });
  selectAll.value = true;
};

const applyAutoGroup = () => {
  if (!segments.value.length) return;
  const raw = segments.value;
  const val = Math.max(1, autoGroupValue.value);
  resetGroupingState();

  if (autoGroupMode.value === 'count') {
    raw.forEach((seg, idx) => {
      if ((idx + 1) % val !== 0 && idx < raw.length - 1) seg.linkNext = true;
    });
  } else if (autoGroupMode.value === 'total') {
    const totalSentences = raw.length;
    const targetGroups = Math.min(val, totalSentences);
    const baseSize = Math.floor(totalSentences / targetGroups);
    const remainder = totalSentences % targetGroups;

    let cursor = 0;
    for (let g = 0; g < targetGroups; g += 1) {
      const currentGroupSize = g < remainder ? baseSize + 1 : baseSize;
      for (let i = 0; i < currentGroupSize; i += 1) {
        const isLastSentence = cursor === totalSentences - 1;
        const isLastInGroup = i === currentGroupSize - 1;
        if (!isLastSentence && !isLastInGroup) raw[cursor].linkNext = true;
        cursor += 1;
      }
    }
  } else {
    let groupDuration = 0;
    for (let i = 0; i < raw.length - 1; i += 1) {
      const seg = raw[i];
      groupDuration += seg.end - seg.start;
      if (groupDuration < val) {
        seg.linkNext = true;
      } else {
        groupDuration = 0;
      }
    }
  }

  showAutoGroupModal.value = false;
};

const openAutoGroupModal = () => {
  if (autoGroupMode.value === 'total' && segments.value.length) {
    autoGroupValue.value = Math.min(autoGroupValue.value || 1, segments.value.length);
  }
  showAutoGroupModal.value = true;
};

const encodeMP3 = (channels: Float32Array[], sampleRate: number) => {
  type Mp3EncoderCtor = new (channels: number, sampleRate: number, kbps: number) => {
    encodeBuffer: (left: Int16Array, right?: Int16Array) => Int8Array | number[];
    flush: () => Int8Array | number[];
  };

  const Mp3Encoder = (lamejs as unknown as { Mp3Encoder: Mp3EncoderCtor }).Mp3Encoder;
  const encoder = new Mp3Encoder(channels.length, sampleRate, 128);
  const buffer: (Int8Array | number[])[] = [];
  const maxSamples = 1152;

  const floatTo16BitPCM = (input: Float32Array) => {
    const output = new Int16Array(input.length);
    for (let i = 0; i < input.length; i += 1) {
      const s = Math.max(-1, Math.min(1, input[i]));
      output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return output;
  };

  const left = floatTo16BitPCM(channels[0]);
  const right = channels[1] ? floatTo16BitPCM(channels[1]) : undefined;

  for (let i = 0; i < left.length; i += maxSamples) {
    const leftChunk = left.subarray(i, i + maxSamples);
    const chunk = right
      ? encoder.encodeBuffer(leftChunk, right.subarray(i, i + maxSamples))
      : encoder.encodeBuffer(leftChunk);
    if (chunk.length) buffer.push(chunk);
  }

  const flush = encoder.flush();
  if (flush.length) buffer.push(flush);
  return new Blob(buffer as BlobPart[], { type: 'audio/mp3' });
};

const buildGroups = () => {
  const groups: Segment[][] = [];
  let current: Segment[] = [];

  segments.value.forEach((seg, idx) => {
    if (seg.selected) current.push(seg);
    if ((!seg.linkNext || idx === segments.value.length - 1) && current.length) {
      groups.push(current);
      current = [];
    }
  });

  return groups;
};

const generatePlayerHtml = (data: { file: string; text: string }[]) => {
  const serialized = JSON.stringify(data).replace(/</g, '\\u003c');
  const head =
    '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>Audio Player</title>';
  const styles =
    '<style>body{font-family:-apple-system,BlinkMacSystemFont,\\\"Segoe UI\\\",Roboto,Helvetica,Arial,sans-serif;background:#f3f4f6;padding:20px;max-width:900px;margin:0 auto}h1{text-align:center;color:#374151;margin-bottom:24px}.card{background:#fff;border-radius:12px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);overflow:hidden}.row{display:flex;align-items:flex-start;padding:16px;border-bottom:1px solid #e5e7eb;transition:background .2s;cursor:pointer}.row:last-child{border-bottom:none}.row:hover{background:#f9fafb}.row.active{background:#eff6ff;border-left:4px solid #3b82f6}.btn{background:#3b82f6;color:#fff;border:none;width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin-right:16px;flex-shrink:0;cursor:pointer}.text{font-size:16px;line-height:1.6;color:#1f2937;margin-top:3px}</style></head>';
  const bodyOpen = '<body><h1>离线音频片段播放器</h1><audio id="player"></audio><div class="card" id="list"></div>';
  const script =
    `<script>const data=${serialized};const list=document.getElementById('list');const audio=document.getElementById('player');data.forEach((item,i)=>{const row=document.createElement('div');row.className='row';row.innerHTML='<button class="btn">▶</button><div class="text">'+item.text+'</div>';row.onclick=()=>{document.querySelectorAll('.row').forEach(r=>r.classList.remove('active'));row.classList.add('active');audio.src=item.file;audio.play();};if(i===0){row.classList.add('active');audio.src=item.file;}list.appendChild(row);});<\\/script>`;
  return `${head}${styles}${bodyOpen}${script}</body></html>`;
};

const generateAndDownload = async () => {
  if (!audioBuffer.value) {
    status.value = '请先上传并识别音频。';
    return;
  }

  isExporting.value = true;
  status.value = '正在打包导出...';

  try {
    const zip = new JSZip();
    const audioFolder = zip.folder('audio');
    const txtParts: string[] = [];
    const htmlData: { file: string; text: string }[] = [];
    const sampleRate = audioBuffer.value.sampleRate;

    const groups = buildGroups();
    groups.forEach((group, index) => {
      const startTime = group[0].start;
      const endTime = group[group.length - 1].end;
      const combinedText = group.map((g) => g.text).join(' ');
      const fileName = `${index + 1}-${startTime.toFixed(2)}-${endTime.toFixed(2)}.mp3`;
      const startSample = Math.floor(startTime * sampleRate);
      const endSample = Math.floor(endTime * sampleRate);
      const channels: Float32Array[] = [];

      for (let ch = 0; ch < audioBuffer.value!.numberOfChannels; ch += 1) {
        const data = audioBuffer.value!.getChannelData(ch).subarray(startSample, endSample);
        channels.push(data);
      }

      const mp3Blob = encodeMP3(channels, sampleRate);
      audioFolder?.file(fileName, mp3Blob);
      txtParts.push(`${fileName}\t${combinedText}`);
      htmlData.push({ file: `audio/${fileName}`, text: combinedText });
    });

    zip.file('transcription.txt', txtParts.join('\n'));
    zip.file('player.html', generatePlayerHtml(htmlData));

    const content = await zip.generateAsync({ type: 'blob' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(content);
    link.download = 'smart-audio-cuts.zip';
    link.click();

    status.value = '导出成功，文件已下载。';
    setTimeout(() => (status.value = ''), 3000);
  } catch (error) {
    console.error(error);
    status.value = '导出失败，请重试。';
  } finally {
    isExporting.value = false;
  }
};
</script>

<template>
  <div class="bg-gray-100 min-h-screen p-4 md:p-8 text-gray-800 font-sans">
    <header class="bg-white shadow-sm rounded-xl p-6 mb-6 flex justify-between items-center">
      <div>
        <h1 class="text-2xl font-bold text-indigo-600 flex items-center">
          <AudioLines class="w-6 h-6 mr-2" />
          智能音频切片工具
        </h1>
        <p class="text-gray-500 text-sm mt-1">上传音频 → AI 识别 → 智能合并 → 导出素材</p>
      </div>
      <div v-if="segments.length > 0" class="flex space-x-3">
        <button
          class="bg-emerald-50 text-emerald-600 hover:bg-emerald-100 px-4 py-2 rounded-lg flex items-center transition font-medium"
          @click="openAutoGroupModal"
        >
          <Sparkles class="w-5 h-5 mr-2" />
          智能分组
        </button>
        <button
          class="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2 rounded-lg flex items-center shadow disabled:opacity-50 transition font-medium"
          :disabled="isExporting"
          @click="generateAndDownload"
        >
          <Download class="w-5 h-5 mr-2" />
          {{ isExporting ? '处理中...' : '导出 ZIP' }}
        </button>
      </div>
    </header>

    <div
      v-if="!audioBuffer"
      class="bg-white border-2 border-dashed border-gray-300 rounded-xl p-16 text-center hover:border-indigo-400 hover:bg-indigo-50 transition duration-300 group cursor-pointer relative"
    >
      <input
        id="audioInput"
        class="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        type="file"
        accept="audio/*"
        @change="handleFileUpload"
      />
      <UploadCloud class="w-16 h-16 text-gray-300 group-hover:text-indigo-400 mb-4 transition" />
      <div class="text-xl font-medium text-gray-600 group-hover:text-indigo-600">点击上传 MP3 / WAV 文件</div>
      <div class="text-sm text-gray-400 mt-2">全程本地处理，音频不出浏览器。</div>
    </div>

    <div v-if="status" class="mb-6 p-4 bg-white border-l-4 border-blue-500 shadow-sm rounded-r-lg flex items-center animate-fade-in">
      <component :is="isLoading ? Loader2 : Info" class="w-5 h-5 mr-3 text-blue-500" :class="{ 'animate-spin': isLoading }" />
      <span class="text-gray-700">{{ status }}</span>
    </div>

    <div v-if="isTranscribing" class="mb-4 bg-white rounded-xl shadow-sm p-3 flex items-center space-x-3">
      <div class="flex-1">
        <div class="h-2 bg-gray-100 rounded-full overflow-hidden">
          <div class="h-full bg-indigo-500 transition-all duration-200" :style="{ width: `${progressPercent}%` }" />
        </div>
        <div class="text-xs text-gray-500 mt-1">
          处理进度：{{ processedChunks }}/{{ totalChunks || '...' }}（{{ progressPercent }}%）
        </div>
      </div>
      <Loader2 class="w-4 h-4 text-indigo-500 animate-spin" />
    </div>

    <div v-if="segments.length > 0" class="bg-white shadow-lg rounded-xl overflow-hidden">
      <div class="flex justify-between items-center p-4 bg-gray-50 border-b">
        <label class="flex items-center space-x-2 cursor-pointer select-none">
          <input
            class="w-5 h-5 text-indigo-600 rounded focus:ring-indigo-500"
            type="checkbox"
            v-model="selectAll"
            @change="toggleSelectAll"
          />
          <span class="text-gray-700 font-medium">全选所有句子</span>
        </label>
        <span class="text-sm text-gray-400">共 {{ segments.length }} 句，当前合并为 {{ groupCount }} 组</span>
      </div>

      <div class="overflow-y-auto no-scrollbar" style="max-height: 70vh">
        <div v-for="(seg, index) in segmentsWithColors" :key="seg.ref.id">
          <div
            :class="[
              'relative flex items-center p-3 transition-colors duration-300 group-transition',
              seg.bgColorClass,
              seg.isEndOfGroup ? 'border-b border-gray-100' : '',
              'hover:brightness-95'
            ]"
          >
            <div class="absolute left-0 top-0 bottom-0 w-1" :class="seg.accentColorClass" />
            <div class="pl-3 pr-4">
              <input
                v-model="seg.ref.selected"
                class="w-5 h-5 text-indigo-600 rounded border-gray-300 focus:ring-indigo-500"
                type="checkbox"
              />
            </div>
            <button
              class="mr-4 text-gray-400 hover:text-indigo-600 transition transform active:scale-95 focus:outline-none"
              type="button"
              @click="playSegment(seg.ref)"
            >
              <PlayCircle class="w-8 h-8" />
            </button>
            <div class="text-xs font-mono text-gray-400 w-24 flex-shrink-0 leading-tight">
              <div>{{ formatTime(seg.ref.start) }}</div>
              <div>{{ formatTime(seg.ref.end) }}</div>
            </div>
            <div class="flex-1 ml-2">
              <div class="w-full p-2 rounded text-gray-700 text-base bg-transparent border border-transparent hover:border-gray-300">
                <p class="whitespace-pre-wrap break-words">{{ seg.ref.text }}</p>
              </div>
            </div>
          </div>
          <div v-if="index < segments.length - 1" class="h-0 relative z-10 flex justify-center items-center">
            <div
              :class="[
                'link-btn w-7 h-7 rounded-full flex items-center justify-center border shadow-sm select-none absolute -top-3.5',
                seg.ref.linkNext
                  ? 'bg-indigo-600 text-white border-indigo-600'
                  : 'bg-white text-gray-300 border-gray-200 hover:border-gray-400 hover:text-gray-500'
              ]"
              :title="seg.ref.linkNext ? '点击断开' : '点击合并到下一句'"
              @click="toggleLink(index)"
            >
              <component :is="seg.ref.linkNext ? Link2 : Link2Off" class="w-4 h-4" />
            </div>
          </div>
        </div>
      </div>
    </div>

    <div v-if="showAutoGroupModal" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div class="bg-white rounded-xl shadow-2xl w-full max-w-lg p-6 transform transition-all scale-100">
        <h3 class="text-xl font-bold text-gray-800 mb-6 flex items-center border-b pb-4">
          <Sparkles class="w-5 h-5 mr-2 text-emerald-500" />
          智能分组设置
        </h3>

        <div class="space-y-6">
          <div class="grid grid-cols-3 gap-3">
            <label
              class="cursor-pointer border-2 rounded-lg p-3 flex flex-col items-center hover:bg-gray-50 transition text-center"
              :class="autoGroupMode === 'time' ? 'border-emerald-500 bg-emerald-50' : 'border-gray-200'"
            >
              <input v-model="autoGroupMode" class="hidden" type="radio" value="time" />
              <Timer class="w-5 h-5 mb-2" :class="autoGroupMode === 'time' ? 'text-emerald-600' : 'text-gray-400'" />
              <span class="font-bold text-sm text-gray-700">按时长</span>
              <span class="text-xs text-gray-400 mt-1">例：每 10 秒</span>
            </label>

            <label
              class="cursor-pointer border-2 rounded-lg p-3 flex flex-col items-center hover:bg-gray-50 transition text-center"
              :class="autoGroupMode === 'count' ? 'border-emerald-500 bg-emerald-50' : 'border-gray-200'"
            >
              <input v-model="autoGroupMode" class="hidden" type="radio" value="count" />
              <ListOrdered class="w-5 h-5 mb-2" :class="autoGroupMode === 'count' ? 'text-emerald-600' : 'text-gray-400'" />
              <span class="font-bold text-sm text-gray-700">按句数</span>
              <span class="text-xs text-gray-400 mt-1">例：每 3 句</span>
            </label>

            <label
              class="cursor-pointer border-2 rounded-lg p-3 flex flex-col items-center hover:bg-gray-50 transition text-center"
              :class="autoGroupMode === 'total' ? 'border-emerald-500 bg-emerald-50' : 'border-gray-200'"
            >
              <input v-model="autoGroupMode" class="hidden" type="radio" value="total" />
              <PieChart class="w-5 h-5 mb-2" :class="autoGroupMode === 'total' ? 'text-emerald-600' : 'text-gray-400'" />
              <span class="font-bold text-sm text-gray-700">按组数</span>
              <span class="text-xs text-gray-400 mt-1">例：目标 20 组</span>
            </label>
          </div>

          <div class="bg-gray-50 p-5 rounded-lg border border-gray-100">
            <label class="block text-sm font-medium text-gray-700 mb-2">{{ inputLabel }}</label>
            <div class="flex items-center">
              <input
                v-model.number="autoGroupValue"
                :max="autoGroupMode === 'total' ? segments.length : undefined"
                class="flex-1 border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-emerald-200 outline-none"
                min="1"
                type="number"
              />
              <span class="ml-3 text-gray-500 text-sm font-medium">{{ inputUnit }}</span>
            </div>
            <p class="text-xs text-gray-400 mt-2">{{ inputHint }}</p>
          </div>
        </div>

        <div class="flex justify-end space-x-3 mt-8">
          <button class="px-4 py-2 text-gray-500 hover:text-gray-700" @click="showAutoGroupModal = false">取消</button>
          <button class="bg-emerald-600 hover:bg-emerald-700 text-white px-6 py-2 rounded-lg font-medium shadow-md transition" @click="applyAutoGroup">
            开始分组
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.no-scrollbar::-webkit-scrollbar {
  display: none;
}

.no-scrollbar {
  -ms-overflow-style: none;
  scrollbar-width: none;
}

.link-btn {
  transition: all 0.2s;
  cursor: pointer;
  z-index: 20;
}

.link-btn:hover {
  transform: scale(1.1);
}

.group-transition {
  transition: background-color 0.3s ease;
}
</style>
