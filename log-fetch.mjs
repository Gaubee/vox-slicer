import { pipeline, env } from "@xenova/transformers";

const originalFetch = global.fetch;
global.fetch = async (...args) => {
  const url = args[0];
  console.log('[FETCH]', url);
  return originalFetch(...args);
};

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = './public/models/';

(async () => {
  console.time('pipeline-load');
  const asr = await pipeline('automatic-speech-recognition', 'whisper-tiny');
  console.timeEnd('pipeline-load');
})();
