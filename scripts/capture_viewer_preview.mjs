/*
 * Capture the public 3-D preview from the real self-contained viewer.
 *
 * One-off presentation tooling: it records the canvas rendered by viewer.html and
 * encodes those frames as a small GIF. It does not generate or alter trajectories.
 * Run with Playwright and gif-encoder-2 available in the capture environment.
 */
import fs from "node:fs";
import path from "node:path";
import { chromium } from "playwright";
import GIFEncoder from "gif-encoder-2";
import { PNG } from "pngjs";

const root = process.cwd();
const viewerPath = path.join(root, "output", "demo", "viewer.html");
const outputPath = path.join(root, "docs", "public", "media", "3d_reconstruction.gif");
const frameDir = path.join(root, ".viewer_preview_frames");

if (!fs.existsSync(viewerPath)) {
  throw new Error("Missing output/demo/viewer.html. Build the real demo pack first.");
}

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.rmSync(frameDir, { recursive: true, force: true });
fs.mkdirSync(frameDir, { recursive: true });

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
await page.goto(`file:///${viewerPath.replaceAll("\\", "/")}`);
await page.waitForTimeout(500);

const captures = [];
const capture = async (count, delay = 120) => {
  for (let i = 0; i < count; i += 1) {
    captures.push(await page.screenshot({ type: "png" }));
    await page.waitForTimeout(delay);
  }
};

// The buttons and playback are the viewer's own controls. The preview therefore
// contains the same court, trajectories, evidence panel and timing rendered by the
// committed HTML artifact.
await capture(6);
await page.locator("#list .seg").nth(1).click();
await page.locator("#play").click();
await capture(28, 100);
for (const view of ["side", "top", "baseline", "broadcast"]) {
  await page.locator(`[data-view="${view}"]`).click();
  await capture(8, 100);
}

await browser.close();

const first = PNG.sync.read(captures[0]);
const encoder = new GIFEncoder(first.width, first.height, "neuquant", false, 10);
encoder.setDelay(100);
encoder.setRepeat(0);
encoder.setQuality(12);
encoder.start();
for (const buffer of captures) {
  encoder.addFrame(PNG.sync.read(buffer).data);
}
encoder.finish();
fs.writeFileSync(outputPath, encoder.out.getData());
fs.rmSync(frameDir, { recursive: true, force: true });
console.log(`Wrote ${outputPath} from ${captures.length} viewer frames`);
