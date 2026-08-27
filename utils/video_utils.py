import cv2
import os

def stub_path_for_video(base_stub: str, video_path: str) -> str:
    """
    Derive a per-video cache path from a generic stub path.

    Detection stubs were keyed to nothing: `tracker_stubs/ball_detections_tracknet.pkl`
    was written by whichever clip ran last and then loaded by whichever clip ran next.
    Analysing clip A and then clip B silently gave B the ball positions of A, and every
    downstream number - speeds, bounces, shot frames - was computed from another video's
    ball. Nothing crashed and nothing looked wrong, which is what made it dangerous: the
    evals in eval/ shared the same stub, so a clip could be graded on detections that
    never came from it.

    Keying on the video's filename stem also lets several clips stay cached at once,
    which is what the multi-clip eval sweeps actually want.

        tracker_stubs/ball_detections_tracknet.pkl
        → tracker_stubs/ball_detections_tracknet__clip_05_wimbledon.pkl

    The stem alone is NOT proof of identity, and the frame-count check is not either.
    Two clips can share a filename and a length:

        matches/2026-01-05/rally.mp4     15 s at 25 fps = 375 frames
        matches/2026-01-12/rally.mp4     15 s at 25 fps = 375 frames

    Those collide on the stem, pass `stub_matches_frames`, and the second clip is then
    analysed with the first one's ball positions. Nothing crashes and nothing looks wrong,
    which is the same silent failure this keying was introduced to stop, one level down.

    So the byte size goes into the key as well. Two different video files sharing a name
    AND an exact byte count is not a case worth engineering against; sharing a name alone
    is ordinary. Callers should still verify with `stub_matches_frames`, which catches the
    remaining case of the same file re-cut to a different length.

        tracker_stubs/ball_detections_tracknet.pkl
        → tracker_stubs/ball_detections_tracknet__clip_05_wimbledon_4615919.pkl

    A video that does not exist keeps the name-only key: it cannot be analysed anyway, and
    this function is called in tests with paths that were never on disk.
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    try:
        stem = f"{stem}_{os.path.getsize(video_path)}"
    except OSError:
        pass
    root, ext = os.path.splitext(base_stub)
    return f"{root}__{stem}{ext}"


def stub_matches_frames(detections, frames) -> bool:
    """
    Cheap sanity check that a cached stub belongs to the clip being processed.

    Detections are one entry per frame, so a length mismatch means the cache came from
    a different video (or a different `--max-frames` run) and must not be trusted. This
    catches the case that filename keying alone cannot: same clip name, different cut.
    """
    return len(detections) == len(frames)


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path, fps=24.0):
    """
    Write annotated frames to disk.

    `fps` must be the SOURCE video's frame rate. It was hardcoded to 24, so a 30 fps
    clip was written 25% slow: the demo video played in mild slow motion, and its clock
    drifted against the 3-D viewer's timeline, which uses the real rate. Every rendered
    output shared that error.
    """
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    height, width = output_video_frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(output_video_path, fourcc, float(fps) if fps and fps > 0 else 24.0,
                          (width, height))
    
    
    if not out.isOpened():
        print(f"ERROR: Could not open video writer for {output_video_path}")
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
        
        if not out.isOpened():
            print(f"CRITICAL ERROR: Failed to create video writer with multiple codecs")
            return False
    
    
    for frame in output_video_frames:
        out.write(frame)
    
    
    out.release()
    
    
    if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
        print(f"Successfully saved video to {output_video_path}")
        return True
    else:
        print(f"WARNING: Video file creation failed or file is empty")
        return False