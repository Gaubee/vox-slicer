import { existsSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

function runGitSubmoduleUpdate() {
  return new Promise((resolve) => {
    const proc = spawn('git', ['submodule', 'update', '--init', '--recursive'], {
      cwd: projectRoot,
      stdio: 'inherit',
      shell: process.platform === 'win32'
    });
    proc.on('close', (code) => {
      if (code !== 0) {
        console.warn('git submodule update exited with code', code);
      }
      resolve();
    });
    proc.on('error', (err) => {
      console.warn('git submodule update failed:', err.message);
      resolve();
    });
  });
}

async function main() {
  const gitmodulesPath = path.join(projectRoot, '.gitmodules');
  if (!existsSync(gitmodulesPath)) {
    return;
  }
  await runGitSubmoduleUpdate();
}

main().catch((err) => {
  console.warn('sync-submodules encountered an error:', err.message);
});
