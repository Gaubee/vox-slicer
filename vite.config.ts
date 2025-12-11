import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { viteSingleFile } from "vite-plugin-singlefile";
import path from "path";

export default defineConfig(({ command }) => ({
  plugins: [
    vue(),
    // 内联 JS/CSS 以生成单文件 HTML
    viteSingleFile({
      removeViteModuleLoader: true,
    }),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  // 开发模式需要本地 public 以访问模型；构建时关闭 public 拷贝，避免 dist 体积暴涨。
  publicDir: command === "serve" ? "public" : false,
  build: {
    target: "esnext",
    assetsInlineLimit: 100000000,
    outDir: "dist",
    rollupOptions: {
      output: {
        manualChunks: undefined,
      },
    },
  },
  worker: {
    format: "es",
  },
}));
