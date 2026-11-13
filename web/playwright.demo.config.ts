import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/demo",
  timeout: 180_000,
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: "http://localhost:5173",
    viewport: { width: 1280, height: 800 },
    colorScheme: "dark",
    channel: "chrome",
    launchOptions: {
      args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"],
    },
  },
  webServer: {
    command: "npm run dev -- --host localhost --port 5173 --strictPort",
    url: "http://localhost:5173",
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
