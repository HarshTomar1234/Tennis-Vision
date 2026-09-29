/* Package the generated viewer without publishing the local annotated broadcast video. */
import fs from "node:fs";

const source = fs.readFileSync("output/demo/viewer.html", "utf8");
const publicPage = source.replace(/"video"\s*:\s*"[^"]+"/, '"video":null');

fs.mkdirSync("docs/public/media", { recursive: true });
fs.writeFileSync("docs/public/media/tennis_vision_3d_viewer.html", publicPage);
fs.copyFileSync("output/demo/viewer_screenshot.png", "docs/public/media/tennis_vision_3d_viewer.png");

console.log("Packaged docs/public/media/tennis_vision_3d_viewer.html");
console.log("Copied docs/public/media/tennis_vision_3d_viewer.png");
