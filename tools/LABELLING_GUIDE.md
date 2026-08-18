# Labelling guide: shot types and events

This is the one task that nothing else can substitute for. Three approaches to
forehand/backhand have been built and measured, and every one was rejected. What is missing
is not a better model, it is ground truth on the footage this pipeline actually runs on.

Roughly 45 minutes of work unlocks: verified shot-type accuracy, volley and smash accuracy
(which have never been checked against labels), a fair test of pose classifiers on broadcast
video, and the ability to say whether the reported forehand/backhand split is right.

---

## The tool

Use `tools/label_shots.py`, which is in this repository. Do not use CVAT, Label Studio or
Roboflow for this: those are built for drawing boxes on images, and this task is picking
frames out of a video and typing one letter. The purpose-built tool also **pre-seeds
candidates from the detector**, so you jump between the frames the pipeline already thinks
are events rather than scrubbing 570 frames blind.

```bash
cd D:/Tennis-Vision
./venv/Scripts/python.exe tools/label_shots.py --video datasets/eval_clips/input_video_5.mp4
```

Output is written to `datasets/labels/input_video_5.csv` automatically. Press `s` to save;
it rewrites the whole file each time, so there is no partial-state risk.

---

## Order to do the clips

Start easy, finish with the hard one. Calibrating your eye on a clean clip first makes the
difficult calls later more consistent.

| Order | Clip | Shots | Why this order |
|---|---|---|---|
| 1 | `input_video_5.mp4` | 11 | Court fits well, no serves claimed. Good calibration. |
| 2 | `input_video_7.mp4` | 8 | Short, one serve claimed. Confirms serve detection. |
| 3 | `input_video_3.mp4` | 7 | Two serves claimed 223 frames apart. Plausible, worth confirming. |
| 4 | `input_video_9.mp4` | 18 | Longest. Two serves claimed. |
| 5 | `input_video_4.mp4` | 5 | Short. Previously reported two serves 18 frames apart, now fixed; check it reports one. |
| 6 | `input_video_8.mp4` | 4 | Shortest. |
| 7 | `input_video_10.mp4` | 11 | |
| 8 | `input_video_11.mp4` | 12 | |
| 9 | `input_video_6.mp4` | 10 | **Do last.** Court fit fails and pose resolves on only 10% of contacts. Expect it to be hard and possibly not worth labelling fully. |

Budget about 4 to 6 minutes per clip once you find the rhythm.

---

## Controls

Every action is a single keypress, no modifiers.

```
n / p        jump to next / previous detector candidate   <- your main navigation
<- ->        step 1 frame          , .    step 10 frames
space        mark CONTACT (a racket hit)
b            mark BOUNCE (ball hits the ground)
1-8          shot type:  1 serve   2 forehand  3 backhand  4 volley
                         5 smash   6 slice     7 drop      8 lob
tab          switch player 1 / 2
u            toggle "unsure" on the label at this frame
x            delete the label at this frame
s            save          q   quit
```

---

## The procedure, per candidate

1. Press `n` to jump to the next candidate the detector proposed.
2. Step `<-` / `->` one or two frames to find the **exact** frame of contact.
3. Press `space` for a racket hit, or `b` for a bounce.
4. Press the shot-type digit if it was a hit.
5. If you are not sure, press `u` as well.
6. If the candidate is not an event at all, **just move on**. Do not label it.

Then sweep the clip with `.` (10-frame steps) looking for events the detector missed
entirely, and label those too.

---

## Where exactly to place the mark

This is the part that decides whether the labels are usable.

**Racket hit.** The frame where the ball is closest to the racket and its direction has not
yet changed. Step back and forth: the contact frame is the last one before the ball reverses.
If two adjacent frames both look plausible, take the earlier one and press `u`.

**Bounce.** The frame where the ball is at its lowest point on screen before rising. Image y
grows downward, so this is the largest y. It is often one frame earlier than it looks,
because the ball is already rising by the time the eye notices.

**Serve.** The contact at the top of the toss, not the toss itself and not the follow-through.
The ball should be above the player's head and the player at or behind the baseline.

---

## Priorities, in order of value

**1. Mark bounces as carefully as contacts.**

This is the least obvious and the most valuable. A volley is defined as "no bounce between
the opponent's contact and this one", so a missed bounce turns a groundstroke into a false
volley. Bounce labels are half the value of this exercise, and the pipeline currently has no
bounce ground truth on broadcast footage at all.

**2. Skipping a bad candidate is data.**

Which candidates you decline to label is exactly the false-positive ground truth the project
has never had. Do not feel obliged to label something just because the detector proposed it.

**3. Label events the detector missed.**

Sweep the whole clip, not just the candidates. Missed events are the recall ground truth,
and recall is currently estimated at 72% with no broadcast confirmation.

**4. Use `u` freely.**

An honest "unsure" is worth more than a confident guess. Unsure labels are excluded from
accuracy denominators rather than counted against the pipeline, so marking them costs you
nothing and protects the numbers.

---

## Judgement calls, decided in advance

Deciding these now keeps the labels internally consistent, which matters more than any
individual call being perfect.

| Situation | What to do |
|---|---|
| Two-handed backhand | `3` backhand. Two-handed forehands essentially do not exist in tennis. |
| Slice or chop | `6` slice if clearly sliced, otherwise `2`/`3`. Consistency matters more than the distinction. |
| Half-volley (ball hit just after the bounce) | `2`/`3`, not volley. A volley is struck **before** the bounce. |
| Ball bounces off the net cord | `b` bounce, and add `u`. |
| Player swings and misses | Do not label. Nothing was struck. |
| Ball leaves frame, you cannot see contact | Do not label, and do not guess. |
| Serve fault, second serve follows | Label both as `1` serve. |
| Cannot tell forehand from backhand | Mark `space`, skip the digit, press `u`. A contact with no shot type is still useful. |

---

## What "done" looks like for a clip

- Every real contact and bounce has a mark
- Every mark you were unsure about carries `u`
- Detector candidates that were not events are left unlabelled
- You pressed `s` and the CSV exists in `datasets/labels/`

A clip is not done when the candidate list is exhausted. It is done when the **video** is
exhausted.

---

## After the first clip, stop and tell me

Send me the first CSV before doing the other eight. I will check that the format, frame
alignment and event mix are what the evals expect, so that a misunderstanding costs one clip
rather than nine.

---

## What each label unlocks

| Label type | Measures |
|---|---|
| contacts | shot-frame recall and precision on broadcast, currently estimated from 7 events on one clip |
| bounces | volley correctness, and bounce recall which has no broadcast ground truth |
| shot types | forehand/backhand accuracy, which three separate approaches have failed to establish |
| serve marks | whether the physical serve test is right, currently only known not to fire on mid-rally clips |
| skipped candidates | false-positive rate, never measured on broadcast |

---

## Output format

```csv
frame,event,player_id,shot_type,confidence,notes
27,contact,1,serve,,
48,bounce,,,,
110,contact,2,backhand,unsure,
```

Plain CSV, one row per labelled frame. Hand-editable if you spot a mistake later.
