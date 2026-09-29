# Public media

These files are presentation artifacts from the real Tennis-Vision demo run.

- `3d_reconstruction.gif` is captured from the generated `viewer.html` while its own
  playback and camera controls run. It contains the actual reconstructed segments and
  evidence panel; no trajectory is drawn separately for the preview.
- `tennis_vision_3d_viewer.html` is the self-contained interactive 3-D scene packaged
  without the local broadcast-video reference. The 3-D view remains fully usable when
  opened directly from a checkout.
- `tennis_vision_3d_viewer.png` is a static fallback for environments that do not render
  animated media.

The source run is the reference clip `input_videos/input_video_2.mp4`. The generated
viewer and summary are assembled by `scripts/build_demo_pack.py`; the preview is captured
by `scripts/capture_viewer_preview.mjs` and packaged by
`scripts/package_public_viewer.mjs`.

To regenerate the presentation assets after producing the demo pack:

```bash
npm install --no-save playwright@1.55.0 gif-encoder-2 pngjs
npx playwright install chromium
node scripts/capture_viewer_preview.mjs
node scripts/package_public_viewer.mjs
```
