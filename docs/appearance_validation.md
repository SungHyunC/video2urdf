# Object-aware appearance: validation on the existing captures

These are development results on the existing phone captures, not a general accuracy claim.
The measured skeleton, collisions and physics remain the output of the original fitting
pipeline; inferred geometry and textures are visual hypotheses.

| Capture | Object/material prediction | Appearance experiment |
| --- | --- | --- |
| iPhone LiDAR laptop | Spark: MacBook, metal base and lid | A laptop base can be visually completed to a rectangle within its fitted bounds, with 1 mm bevels; collision geometry is unchanged. Visible screen, keyboard, trackpad and cover use video texture. On the separately refitted keyboard-only base, the completion covered 93.7% of the base mask versus 92.7% before, on 35 sampled frames; the lid stayed at 91.7% on 60 sampled frames. The underlying keyboard-only base is known to be incomplete. |
| iPhone LiDAR drawer | Spark: three-drawer cabinet with handles, wood exterior | The measured moving drawer and its fitted tray are retained. Two closed fronts and handles are added as visual-only hypotheses, evenly spaced using the fitted cabinet bounds. Mass, inertia, joint ranges and URDF collision mesh references match the original. On 796 frames, quarter-resolution mask coverage is 54.8% for the drawer and 62.5% for the base, the same as before completion; the fit still needs review. |
| RBO book03 | Spark: hardcover book, paper page block | A rectangle inferred from the visible cover face alone reduced coverage on 21 sampled frames from 90.0% to 78.9%, so it was rejected. Blending it 50% toward the fitted silhouette raised sampled coverage to 87.2%, allowing it through the visual guard. On the full video at quarter resolution, the resulting lid covers 89.9% on 114 usable frames (original 92.2%), while base coverage is 63.6% on 21 frames. It still needs review and is not an independently validated reconstruction. |

Spark's count, material and unseen surface details can be wrong. New fronts are fixed visual
details, not additional measured joints. The fitted drawer's hidden depth and panel
thickness include assumptions. The ordinary RGB-only MapAnything path has a much less
reliable metric scale; one known length corrects global scale but not local depth or focal
error.

As a calibration plumbing check, a six-frame RGB clip was reconstructed with MapAnything.
Using two arbitrary pixels and a hypothetical 100 mm reference changed their reconstructed
distance from 384.90 mm to 99.81 mm after the depth and camera translations were rescaled;
the resulting capture loaded successfully. This is not a real-world dimension accuracy test.

On the full laptop video using the same manually corrected tracks and reference frame 833,
an existing RGB-only MapAnything capture with an applied 0.252 global scale produced a
reference opening of 84.3° and a preliminary lid of 222.5 × 107.5 mm; its fitted base
thickness was 78.4 mm. The LiDAR-based comparison with those tracks produced 107.2° and
305.6 × 206.4 mm, versus an approximate lid specification of 304 × 212 mm. A second
existing RGB capture using the supplied 1385 px focal length and scale 0.276 produced
a reference opening of 1.8° and plate span of 256.2 mm. Both RGB fits are inaccurate;
the scale factors were already present in those captures and are not independent
reference measurements. Do not infer millimetre accuracy from the calibration plumbing test.
At reference frame 833, matching pixels between the LiDAR and RGB-only depth maps gave
median signed RGB-minus-LiDAR depth errors of +32 mm on the floor, -47 mm on the combined
base and -41.5 mm on the screen-track region (20,135, 6,128 and 8,780 pixels respectively).
The differently signed local errors explain why a single global scale cannot correct the
floor-to-base separation in this example; LiDAR itself also has measurement noise.

The appearance stage compares each new visual against the pre-completion model at up to
60 sampled frames. It retains the fitted visual if fewer than five usable masks exist or
mean coverage or, for full-part masks, mean mask IoU drops by more than five percentage points.
IoU is recorded but not used as an excess-shape gate for known partial masks such as a
keyboard or an explicitly combined set of base regions; that limitation is reported. The checked masks are still SAM
predictions, so passing this guard cannot establish physical dimensions or hidden shape.
For drawer models the tracked fronts and cabinet panels are also partial-link masks, so
their IoU is reported without treating the rest of the 3D link as a false positive.
For a book, if the first inferred rectangle fails this guard, it tries progressively more of
the fitted silhouette and records the selected blend and each comparison.

## Three independent Spark runs on the same laptop capture

These are the last fits in `data/repeat/run1`, `run2`, and `run3` (same capture; no manual
intervention). The reference lid size is approximately 304 × 212 mm. Mask coverage is the
fraction of each tracked SAM mask covered by the replay, not IoU against independent ground
truth. Values below are rounded from the run output; they are not independent validation of
the masks.

| Run | Lid size (mm) | Replay coverage: lid / base | Notable limitation |
| --- | --- | --- | --- |
| 1 | 316.3 × 216.2 | 75% / 86% | Lid thickness fell back to 6 mm; plate track drifted onto the closed lid. |
| 2 | 304.5 × 215.1 | 93% / 16% | Base thickness fell back to 12 mm; base mask and fitted base disagree. |
| 3 | 304.3 × 213.6 | 93% / 31% | Lid thickness clipped to 1 mm; base reads 36.6 mm, inconsistent with the laptop. |

The lid width ranges 304.3–316.3 mm and depth 213.6–216.2 mm. All three runs produced a
model; only run 1 clears a 70% coverage threshold for both links, and it still has a known
lid-tracking and thickness failure. Do not report this as three reliable reconstructions.

After those runs, reference-frame selection was changed to prefer a sufficiently visible
base and reject frames where the base mask contains most of the plate. Refitting the *same*
run 2 and 3 tracks selected frames 821 and 919 instead of 684 and 685. On six sampled base
frames at quarter resolution, run 2 coverage increased from 9–30% to 74–97%, and run 3
from 21–37% to 94–97%. Over all 923 frames at quarter resolution, using the same tracked
masks, run 2 base coverage rose from 15.6% to 86.0% and run 3 from 31.1% to 95.0%; lid
coverage was 94.5% and 92.3% after refitting. Base coverage is computed only while the
open-indicator mask is usable (523 and 524 frames). This is not a new independent
success-rate test: the run 2 refit also produced a suspect 24 mm lid thickness. Thickness
handling still needs review. Run 2's reference frame has no usable floor mask; its 12 mm
base thickness is a fallback and is now recorded as unmeasured.

Reference selection now additionally requires a tracked open-indicator mask in the chosen
frame when that role is provided, and fails the fit if no suitable frame remains. Applying
this selection to the same three stored tracks chooses frames 803, 821 and 919 respectively;
this check alone does not validate that the indicator itself was correctly identified.

A later independent Spark execution confused the open screen and closed-lid tracks: its
reference frame 803 was visibly open, but the selected plate mask covered the base and the
fit reported 0.3° and a 287 × 67 mm lid. On a separate copy of those tracks, swapping the
two plate roles produced reference frame 833, opening 107.2°, and a preliminary lid size
305.5 × 206.4 mm (the approximate specification is 304 × 212 mm). This is a manual
diagnostic using the same capture, not a new autonomous success-rate result. Full-resolution
replay covers 91.1% of the lid mask on 400/923 frames and 94.0% of the base mask on 523/923
frames; coverage is not independent ground truth. The base mesh spans only about 112 mm in
one direction because its mask captures the keyboard rather than the entire base. In the
reference frame, combining tracked keyboard and touchpad depth points increases their
1st–99th percentile span in that direction from 105 mm to 192 mm, still excluding unseen
margins. This is a diagnostic, not a validated full-base reconstruction. The agent is now prompted to test a role
swap when the masks show this particular overlap; it must still check the replay.

A further local experiment builds the base from the union of tracked keyboard and touchpad
regions on frames where the keyboard is visible. Its preliminary base mesh extent is
approximately 290 × 199 × 15 mm, versus 290 × 112 × 16 mm with the keyboard alone.
This uses the previously fitted hinge and remains an experiment: full-resolution replay
coverage is 91.2% of the lid mask on 400/923 frames and 94.6% of the combined base mask on
523/923 frames. The masks are not independent ground truth, and the unobserved base margins
remain unvalidated. The agent can now combine separately tracked rigid regions with a
visibility gate; this capability has not yet been tested in a new autonomous run.
On a copy of that combined-base model, visually filling the base corners within its fitted
bounds changed sampled base-mask coverage from 93.3% to 93.8% on 35 frames; lid coverage
remained 91.7% on 60 frames. This coverage check does not assess excess silhouette or
establish the true unseen outline. The completed visual and tracked-union provenance were
exported to a bundle that loads in MuJoCo.
The same comparison's base mask IoU fell from 74.6% to 62.3%; since the mask is only the
keyboard and touchpad, it does not describe the full laptop surface. The report records this
score and warns that excess visual shape cannot be validated from that partial mask.

On the corrected combined-base fit, SAM 3D at reference frame 833 generated a base with
1.0 mm median and 2.3 mm p90 observed-depth distance, but its 295 × 210 × 25 mm extent
failed the fitted-size gate. Its lid candidate was farther from observed depth (78 mm
median, 161 mm p90) and also failed the extent gate. Neither was activated.
The manually re-anchored fit was also tested at frame 831. Its SAM 3D base candidate
had 1.0 mm median and 3.3 mm p90 depth distance but an extent of about 354 × 272 × 27 mm;
the lid candidate had 25.6 mm median and 78.2 mm p90 depth distance and extent about
322 × 249 × 236 mm. Both failed the size/depth gate and retained their fitted visuals.

Another independent agent run selected reference frame 922 with the screen correctly
segmented there, but the previous frame's screen track had drifted onto the base. The
temporal outlier filter removed frame 922 and the old implementation silently computed
its reference opening from frame 921, giving 0.3° despite an open laptop. The hinge fit
now rejects a missing or filtered reference angle and asks for corrected tracking; this
does not itself repair the track. The already-running agent used the earlier implementation.
On those same tracks, rejecting a missing-plate fallback to the visible open base and
discarding 36 additional open-frame masks that strongly overlap the base still left the
reference angle rejected. Re-anchoring/retracking remains necessary for that run.
In a later independent run the selected reference frame 817 had no tracked floor. The
nearest floor frame with at least 50 depth points was before it; a diagnostic over
frames 580–625 gave approximate base-to-floor separations of 9–16 mm, despite camera
pose and floor leveling. A single nearby frame is therefore not a dependable thickness
measurement in this example, and the fitted 12 mm fallback remains marked unmeasured.

The agent now has a `retrack_part` tool to re-anchor one original part while preserving
the other tracks; changing that track invalidates the prior fit. A manual screen-only
retracking experiment used nine anchors, seven newly segmented on open frames. On the
523 frames where the keyboard was visible, the number where more than half the screen
mask overlapped the base fell from 221 to 10. A refit selected frame 831 and reported
106.8° opening and 304.7 mm hinge span; the original run had reported 0.3°. This is a
manual repair on the same capture, not an independent autonomous success-rate test.
The resulting full-resolution replay covers 91.6% of the lid mask on 612/923 frames
and 95.5% of the base mask on 523/923 frames. The fitted lid is 305.6 × 207.9 mm and
5.1 mm thick, while the base mesh's axis-aligned extent is approximately 314.5 × 228.1 mm.
These mask-based scores are not independent ground truth; unscored frames and remaining
bad masks still require inspection.
When a reference angle is rejected, `fit_lidar` now proposes up to seven frames spread
across the available open-indicator track and points the agent to `retrack_part`. These
are candidate frames, not verified segmentations; the agent must inspect each one.
Overlap warnings now focus on frames where the open indicator is available: the original
run still triggers a drift warning, while the repaired track no longer triggers the old
warning that counted expected closed-frame overlap. The repaired artifact includes
`repair.json` and is explicitly marked as manually repaired.
To check the tool's integration without another GPU call, the actual returned screen-track
records were replayed through `retrack_part` on a copy of the old run. The output matched
the manually repaired screen on frames 0, 150, 750, 900, 921 and 922; the floor, base and
keyboard tracks were byte-for-byte identical after decoding on all of their frames. This
checks track replacement, not the agent's autonomous choice of anchors.
On a separate copy of the repaired model, physics was recomputed with the prior agent's
estimated 0.9 kg base and 0.35 kg lid masses, contact friction 0.4, and its estimated
hands-off intervals of 0–2 s and 9–11 s. The resulting joint-friction estimate was
0.200 N·m. In the illustrative release video, the model moved about 0.1° with that value
and 17° without it. These inputs and the manual tracking repair are recorded in the
artifact; neither the masses nor hands-off intervals were independently verified.
An earlier sampled base IoU of 0.41 incorrectly included closed frames where the tracked
base mask also contained the lid. Restricting the comparison to 35 sampled frames with a
usable open indicator gives 0.65 before and 0.64 after visual completion. This better
matches the base-only evaluation, but the masks are still not independent ground truth.

An additional independent agent run selected separate screen, base, closed-object and
keyboard tracks without manual intervention. Its initial fit used reference frame 817,
reported a 106.0° reference opening and 304.5 × 214.0 mm lid, and covered 92.8% of the
tracked lid mask on 567/923 frames and 93.2% of the base mask on 523/923 frames. The
12 mm base thickness was a fallback and the 1 mm lid thickness was clipped; neither is
a reliable measurement. The run completed without manual intervention; in 523 open
frames with usable base masks, no screen mask overlapped the base by more than 50%.
SAM 3D candidates for both links failed the extent gate and were rejected. The result
therefore remains marked for review, and one run does not establish a success rate.

Texture projection now uses the rendered visibility of each individual geometry together
with its own tracked mask, so partial occlusion does not paint another part onto a face.
On this run, two lid-shell faces received video texture, including the visible logo;
unseen faces still use a fallback appearance.

## SAM 3D Objects

Access was approved and the real checkpoints were tested on the same two captures with
depth pointmaps. Early candidate meshes often represented the visible panel but not the
whole articulated link. For example:

* Laptop lid at frame 850: observed points to mesh median 26.2 mm, p90 54.7 mm before
  registration. At frame 0, bounded rigid registration reduced the median from 14.0 to
  7.1 mm (p90 25.0 mm), but its extents were about 328 × 187 × 285 mm in link coordinates,
  versus fitted 310 × 63 × 219 mm.
  A further attempt using the closed-laptop mask at frame 450 was worse: median 78.5 mm,
  p90 103.9 mm. That mask may include the base, so it is exploratory rather than a clean
  measurement of the lid alone.
* Drawer front at frame 0: observed points to mesh median 2.1 mm and p90 4.5 mm, but
  the generated extent was about 52 × 468 × 207 mm versus the fitted drawer link's
  248 × 327 × 122 mm. Good agreement on the observed front did not mean the hidden drawer
  was reconstructed correctly.
* RBO book03 cover, fixed RGB-D camera with tracked object motion: observed points to
  mesh median 33.0 mm and p90 43.2 mm; extents 244 × 56 × 176 mm versus the fitted link's
  395 × 75 × 275 mm. The fitted link is itself an estimate, not independent ground truth.

These candidates were **not activated**. The current acceptance gate checks both observed
depth distances and overall fitted extents; rejection retains the fitted visual. Thresholds
are engineering safeguards, not a guarantee of accuracy, and the fitted extents can include
assumptions. The generated candidate files remain in the ignored `data/appearance_trials/`
directory for inspection. No diagnostic non-uniformly stretched mesh is used by the pipeline.

### Alternate SAM 3D frame experiment

On the independent autonomous laptop fit, a separate SAM 3D lid candidate was generated
from open frame 150 with seed 43, rather than frame 817 with seed 42. Its observed-depth
median and p90 distances were 0.61 and 2.26 mm; its 299.5 × 71.4 × 227.4 mm extent
passed the existing 80–120% fitted-extent gate (fitted 313.2 × 64.7 × 221.0 mm).
This proves that one candidate passed those geometric checks, not that hidden surfaces
are correct. In a rendered turntable its screen appears nearly black and loses the
source screen image, so the geometric gate alone does not establish appearance quality.
The candidate is kept in a separate `sam3d_alternate_open` result for comparison; the
original fitted-and-video-textured result remains available.

A trial projected the video-textured screen mesh onto generated vertices within 4 mm,
transferring 14,331 of 30,008 vertex colours. The turntable showed large discontinuous
black regions where the candidate surface was farther away, so the transfer was not
retained or enabled. A robust solution needs a texture mapping that accounts for the
actual generated surface and occlusions, not just nearest-plane distance.

Increasing the texture-transfer distance to 25 mm covered 19,596/30,008 generated
vertices and removed much of the front gap, but a fixed-pose rear view showed the screen
image projected onto the back as well. This experiment is kept separately under
`sam3d_texture_trial` and is not enabled in the pipeline. SAM 3D geometry acceptance
now explicitly requests visual review of texture and hidden surfaces.

A subsequent visibility-gated projection transferred 10,773/30,008 vertices from the
source screen texture only where the SAM 3D surface was visible in frame 150. The front
screen was mostly restored, and the newly transferred landscape did not appear on the
back in a fixed-pose rear rendering. However, the rear still showed a different screen
image from SAM 3D's own generated vertex colours: visibility gating the transfer alone
does not reconstruct the lid exterior. The trial is not enabled automatically, and its
25 mm transfer radius is a visual interpolation tolerance, not a measurement.

A further manual trial used frame 343, where the closed lid exterior is visible, to
project the existing lid texture onto the visible generated surface, after the frame 150
front transfer. It transferred 10,718 vertices on the second pass; fixed-pose front and
rear renderings show the screen and Apple logo on their respective sides. The edges still
contain generated colours and artifacts. The 25 mm projection limit is a visual tolerance,
not measured accuracy; this manually chosen two-frame procedure is not yet a general
automated solution. The separate `sam3d_two_sides` viewer result records this provenance.

An experimental `--auto-views` transfer now uses recorded observed faces and their
source frames from `texture.json`; unobserved source faces are excluded. On the same
accepted SAM 3D laptop candidate it chose frame 828 for the screen and frames 343 and
531 for the lid, transferring 11,292, 10,412 and 9,370 vertices respectively. A fixed
open-pose rendering showed the screen on the front and logo on the back, with some edge
artifacts remaining. This is one example with a pre-existing accepted candidate, not an
independent end-to-end success or a guarantee for other objects.

The `--sam3d-video-texture` option is now wired from `video_to_model.py` through
`run_agent.py` into SAM 3D generation. On a copy of the previously fitted laptop run,
a fresh remote SAM 3D lid generation at manually chosen frame 150 and seed 43 passed
the geometric gate, then automatically transferred the recorded screen and lid textures.
Front and rear renders showed the screen and logo, with edge artifacts. This exercised
the SAM 3D and texture stages together, not a fresh video-to-model run; frame selection
for SAM 3D itself was manual.

An automatic one-retry SAM 3D test on the same laptop selected reference frame 817,
then frame 0. Frame 817 failed with 14.0 mm median / 35.1 mm p90 observed-depth
distance. Frame 0 improved to 0.82 mm / 2.70 mm, but its 303 × 82 × 228 mm extent
failed the existing extent gate; the fitted visual remained active. This validates the
retry and rejection path, not automatic recovery. More frames and other objects need
testing; additional retries incur remote inference.

After restricting laptop lid retries to frames with an open indicator and avoiding
the video endpoints for the first retry, an independent SAM 3D invocation on the same
existing fit rejected frame 817 and automatically selected frame 184. The second candidate
passed: depth median 0.86 mm, p90 2.37 mm, extent 305.6 × 71.9 × 231.2 mm versus
fitted 313.2 × 64.7 × 221.0 mm. Video texture transfer then ran automatically. Fixed-pose
front/back renders showed the screen and logo with residual edge artifacts. This is one
retry on an existing LiDAR fit, not an independent end-to-end video run or a success rate.

On a copy of the existing drawer fit, the automatic SAM 3D one-retry flow selected
frames 0 and 795. Both failed: observed-depth median/p90 were 86.0/103.2 mm and
63.9/75.8 mm respectively, and both candidates also failed fitted-extent checks.
The fitted visual remained active. Existing replay coverage was 55% for the drawer and
63% for the base, so the underlying fit itself also requires review. This is evidence
of a failure case for the generalization of the retry method, not an end-to-end run.

Inspection of the original drawer-front tracking mask gives direct evidence of drift:
on frame 30 it extends beyond the closed cabinet onto the floor, and on frame 240 it
collapses to a small fragment. These two observations do not explain every low-coverage
frame, but they show that the masks themselves cannot all be treated as ground truth.
The drawer fit tool now gives the agent a warning for low replay coverage so it can
inspect tracking and refit; the warning does not by itself repair or validate a track.

The simplified laptop visual now completes both base and lid as straight-edged panels
from the fitted geometry and visible inner face, preserving fitted collision geometry.
On a copy of the interrupted independent fit, using an existing laptop classification,
the mask comparison accepted the visual: sampled base coverage 86.2% to 88.7% and
lid coverage 73.7% to 73.7%. A render looks like a simple laptop, but the dark screen
area still follows an irregular fitted mask, and tracking/thickness warnings remain.
This is not an end-to-end independent classification test.

The simplified laptop screen visual is now a straight-edged panel reconstructed from the
fitted screen bounds. On the same interrupted-run fit, mask comparison accepted the
inferred model and a local render showed straight outer and screen edges. This test reused
an existing object classification; it does not establish classification or geometry
quality on arbitrary video. The `laptop_simple` viewer artifact exposes this trial.

The untextured drawer model was inspected from the front: the inferred outer cabinet
and three fronts are recognizable as a drawer cabinet, though panel sizes are uneven and
tracking coverage remains low. When the classifier cannot estimate a drawer count, the
visual template now assumes three and records that assumption in `shape_inference.json`
and `quality.json`; it does not invent extra joints or measurements. A test on a copy of
the existing drawer fit produced two assumed closed fronts alongside the moving one.

A fresh open-weight Muse Spark classification was run on the existing laptop and
drawer videos without reusing `appearance.json`. It identified `laptop` and a three-drawer
cabinet respectively; simplified appearance processing then ran on copies of previously
fitted models. The drawer copy already contained inferred fronts, so no new baseline
comparison was available, and both fits retain their existing tracking warnings. This
checks classification and shape completion on two existing captures, not a fresh
end-to-end fit or generalization to arbitrary objects.

Object-specific visual templates now check the fitted joint type: a laptop/book
template is not applied to a slider-only fit and a drawer template is not applied to a
hinge-only fit. A mismatch remains flagged for review and retains the fitted visual.
A test on a copy of the drawer model with an intentionally wrong laptop classification
confirmed the model XML stayed unchanged; this is not a classification-accuracy test.

An experimental `box` appearance category builds a rectangular base and lid from a
pre-existing hinged fit, keeping collision geometry intact. A synthetic hinged-box test
checks its structure. Two old cardboard-box fits were also inspected: one has invalid
negative base thickness and one has a positive estimate, but neither supplies the replay
data needed for the current end-to-end appearance check. Direct rendering of these older
fits did not establish a satisfactory recognizable model. Invalid base thickness now
prevents box template completion; real-video classification, fitting and visual quality
remain unverified for this category.

A fresh end-to-end run (`simple_e2e_960`) of the laptop RGB-D capture used 960 px SAM
tracking, open-weight Muse Spark, no SAM 3D and no video texture. It completed without
manual intervention, classified the object as a laptop, and activated the rectangular
base, lid and screen visuals. Fitted lid dimensions were 305.6 × 206.8 mm; sampled
rendering is recognizable as a simple laptop, though the base has no keyboard detail.
Replay coverage was 97.5% for the lid on 189/923 frames and 90.5% for the base on
523/923 frames, using tracked masks rather than independent ground truth. The agent
reported removal of 218 open frames where screen and base masks overlapped, and use of
a closed-surface fallback on 516/923 frames. Masses and hands-off intervals were model
estimates, not independently verified. This one LiDAR capture does not validate ordinary
RGB-only scale or other object types.

On the completed `simple_e2e_960` laptop fit, simple appearance now adds a schematic
keyboard panel and trackpad within the fitted base bounds. A render was inspected and
the markings make the model more readily identifiable as a laptop. Their positions and
sizes are visual hypotheses, not measurements. Automated tests verify that these are
visual-only, preserve mass/inertia, are idempotent, and are removed when video textures
are requested.

On the existing RGB-only laptop fit, the simple visual had a visibly over-thick base.
A visual-only prior now limits the base thickness when it exceeds 15% of its smaller
footprint, while keeping fitted collision geometry and recorded dimensions unchanged.
The observed-mask comparison accepted this visual on that fit, but the fitted collision
is still about 78 mm thick; the visual floats above the physics-demo floor and the lid
still overhangs the base. An additional attempt to clip the lid to the base width looked
more regular but failed the observed-mask gate and was not adopted. This remains an
experimental appearance improvement, not a corrected RGB reconstruction.

A follow-up RGB-only trial extended the partial base visual toward the fitted lid width.
Although intended to make the laptop silhouette more coherent, the combined visual failed
the existing observed-mask comparison for the lid and was rejected. This experimental
widening is not enabled. Schematic keyboard and trackpad placement now follows the active
visual base bounds when available.

A manual RGB-only refit attempted frame 150 after mask cleanup. The hinge solver could
compute an angle from a fallback mask, but `build_model` then failed because the actual
plate mask at that frame had been removed. The solver now requires the actual plate mask
at the reference before fitting, so the agent can request re-anchoring rather than fail
later with a KeyError. A separate frame 148 refit has a real plate mask and is running;
its preliminary size and base thickness remain inaccurate, so it is not a validated fix.

The hinge fitting tool now retries up to two other open reference frames when the
selected reference's angle is unavailable, and stores the attempted frames in model
metadata. On the existing RGB-only tracks, reference 833 failed and frame 148 succeeded.
Each retry restores the original camera transforms and repeats floor alignment; this
reproduced the separate frame-148 opening of 124° rather than the 104° produced when
rebasing already-aligned transforms. The 124° estimate and 60 mm base thickness remain
inaccurate compared with the LiDAR fit; this is recovery from a fitting failure, not
validation of RGB metric depth or an independent end-to-end run.

On the RGB-only fit re-anchored at frame 148, the simple laptop remains recognizable,
but the screen overhangs the base and the visual floats above the physics floor because
the inferred base thickness is smaller than the retained collision geometry. A manual
trial clipping lid/screen to the fitted base width improved their outer alignment but
reduced sampled lid mask coverage from 77.1% to 67.5% (IoU 0.517 to 0.478), failing the
existing visual comparison. The candidate was not enabled. These are SAM-mask comparisons,
not independent ground truth; reducing the overhang remains unresolved.

A manual cardboard-box tracking trial used new lid anchors at frames 120 and 160.
The newly tracked base was usable on only 5 frames, versus 34 for the existing base,
so it was rejected. Combining the new lid with the existing base increased the frames
with usable base, lid, and inside masks from one to three. A local refit then completed,
but covered only 45% of lid masks and 86% of base masks (the latter on only three
frames); it estimated a 217 × 83 mm lid and a -167° to 94° hinge range. Those results
are not sufficient to validate the box shape or motion. This was a manually anchored
trial, not an automatic video-to-model success; the generated trial artifacts are not
promoted as a working model.

The moving-drawer handle was also tested through the appearance workflow on a copy of
an existing three-drawer LiDAR fit, reusing its video classification. The generated
handle is attached to `drawer1`, and the 60-frame visual comparison accepted the result:
tracked-mask coverage remained 62.5% for the base and 54.8% for the moving drawer. The
masks only represent selected regions, so this comparison cannot validate the handle's
shape or position. The low coverage and existing tracking drift remain unresolved; this
is not a fresh end-to-end run.

After changing missing drawer tracks to remain unmeasured within a motion window, the
existing three-drawer LiDAR capture was refitted locally using its original tracks and
window (frames 190–650). The refit completed: 445/461 window frames have joint values,
16 are missing, and the 99th-percentile travel was 248 mm. The prior fit has the same
16 missing frames and values in this interval, so this example does not demonstrate a
change or tracking improvement. Replay coverage remains approximately 62% base / 55%
drawer; the capture and manual-agent track selection are not a fresh autonomous run.

A manually selected open-frame drawer front was retracked at 960 px on the existing
three-drawer LiDAR video. Visual inspection of frames 100, 190, 240, 300, 400, 500,
600, 650 and 700 showed the new track follows the operated front more consistently
around opening; the old track is on another front at frame 190 and fragments at 240.
The fit initially failed because the base had no upright plane on the largest-mask
frame; the fitter now tries additional observed times. It completed on the new track,
with joint estimates on all 461 frames of the specified operating window and replay
coverage 62% base / 60% drawer. The previous fit's coverage was 62% / 55%, but the
tracked masks differ, so these numbers do not establish an accuracy improvement.
On replay images the new moving front misses part of the observed front; a simplified
render has a recognizable three-front cabinet but uneven panels. This remains a manual
tracking experiment, not an autonomous end-to-end validation.

On a copy of the manually retracked drawer fit, a visual-only moving front was aligned to
the cabinet slot it overlaps and the same width as the other inferred fronts. Its handle
is attached to the moving link. A front render shows three more even panels; the original
collision, mass, and motion remain unchanged. On 60 sampled frames, moving-link mask
coverage rose from 59.7% to 66.4% while mask IoU fell from 0.427 to 0.407; the base
coverage stayed at 62.5%. The mask is partial and cannot establish whether the added
outer area is correct. These are not measurements of the new edges or a fresh autonomous
run, and the underlying fit still needs review.

An independent open-weight run on the same drawer2 LiDAR video first selected a closed
anchor on a different front and an open anchor on the moving front. On inspected frames
190 and 240 the resulting moving track remained on the neighbouring drawer or a fragment.
The first fit treated a moving track as part of the base and estimated only 8 mm travel,
with 64% base / 22% drawer mask coverage. After the warning, the agent excluded the other
fronts from its base and refitted: travel became 245 mm and coverage 62% / 54%; the
moving track had not yet been corrected. It then requested a new track with similar closed
and open anchors. To reduce this failure in later runs, the drawer instructions no longer
require a closed-frame anchor when the moving front's identity is uncertain. This prompt
change was made while the independent run was already in progress and did not affect it.

That independent agent's requested retrack completed, but visual inspection of its
result still showed the neighbouring front at frame 190 and only a fragment at frame 240.
It subsequently began another fit while reintroducing `drawer_fixed2` into the base,
even though the earlier diagnostic warned it moved. The autonomous repair is therefore
not yet demonstrated; the ongoing fit must not be counted as a successful run merely
because its track has detections on all frames.

A pre-fit duplicate-mask check now samples the drawer's operating window and asks the
agent to inspect tracks before fitting when a base component and moving component occupy
nearly identical masks repeatedly. It flagged `drawer_fixed2` versus `drawer_moving` on
8/40 sampled frames in this independent run. It did not flag the side-panel/moving-front
pairs in either existing drawer fit tested. Overlap is an indicator of possible identity
confusion, not proof of motion; this check has not been validated across unrelated videos.

The third automatic fit reintroduced `drawer_fixed2` as base after retracking and
reproduced the first failure: 8 mm travel and 22% moving-mask coverage. The agent then
requested a new full track with different fixed-front candidates. This illustrates why the
new pre-fit conflict check needs independent validation; the in-progress agent was started
before that check and does not use it.

The independent drawer run reached a fourth fit with about 52% drawer coverage, but its
physics call failed because the agent supplied the tracked part name `drawer_moving`
instead of the generated link `drawer1`. It also stopped before appearance completion
because its classification response was invalid JSON. Appearance inference was resumed
separately and classified the video as a three-drawer cabinet; this is not an uninterrupted
end-to-end success. The classifier now retries one invalid JSON response.

The physics tool now maps an unambiguous tracked drawer name from `slides.json` to its
corresponding generated link and rejects duplicate inputs for one link. On a copy of that
run, the agent's original mass inputs then applied successfully; gravity did not load the
slide, so friction was estimated, not measured. This manual replay validates the name
mapping, not the accuracy of the masses, friction, or drawer reconstruction.

The current RGB-D modelling agent and appearance pipeline still require a fitted hinge or
sliding joint and associated part roles. They do not yet provide a general single-rigid-
object path for an arbitrary static object; `other` retains the existing fit rather than
constructing a new category-specific shape. The laptop and drawer examples should not be
interpreted as proof that arbitrary photographed objects are supported.

A preliminary static-object geometry primitive now fits an upright rectangular envelope
to observed 3D points using a supplied gravity or floor normal. It trims outliers and
reports near-square horizontal orientation as uncertain; it cannot recover unseen surfaces.
Synthetic rotated-box and degenerate-depth tests pass. As a smoke test only, the existing
closed-box track at frame 74 produced observed bounds 210 × 102 × 104 mm using a floor
plane. This is a single selected frame and mask, with no independent dimension check;
the primitive is not yet wired into the video-to-model agent or classification/export.

The static envelope primitive can now generate a single-link URDF/MJCF and replay from
existing RGB-D tracks. A smoke test on a copy of the cardboard-box capture completed and
reported 88.8% mean coverage of the tracked whole-box mask over 222 frames. However the
largest mask selected reference frame 138, where the box lid is open and moving, so the
result is not a valid reconstruction of one static rigid object and must not be promoted.
No independent ground-truth geometry was checked. This builder is not yet wired to
agent classification, static-object detection, or the standard export flow.

The static builder is now exposed to the agent only for a stationary roughly box-shaped
object and is compatible with the appearance and viewer exporters. For a smoke test, the
first 24 frames of the existing cardboard-box clip were extracted along with matching
depth and existing tracks; the camera and object are stationary during this short segment.
The box envelope replay covers 95.1% of the existing tracked mask, and a browser render
shows a straight box. The classifier returned category `box` but used the internal link
name `base` as its object name; the viewer artifact substitutes the category and preserves
the raw response. The workflow now performs the same fallback for that specific failure.
These masks were not newly selected by an autonomous agent, the clip is short, and there
is no independent dimension ground truth. The envelope does not reproduce the box opening,
colours, details or unseen surfaces and does not support arbitrary non-box objects.

A fresh agent run on the 24-frame stationary clip independently segmented the box and
table, tracked them, selected `fit_static`, applied estimated physics, and produced a
viewer. The model's tracked-mask replay coverage was 93.4% over 24 frames. Its initial
classification returned category `box` but a malformed long object-name string; separately
rerunning appearance produced the material word `cardboard` instead of an object name.
The viewer artifact uses `box` as the category fallback and retains the raw response. The
runtime now rejects malformed long names and substitutes the category for a bare material
or internal link name. A missing static-envelope warning was also fixed. This is one very
short clip from an existing source, not a representative success rate or an independent
measurement of hidden dimensions; only a rectangular static object was exercised.

Before creating a static envelope the builder now compares sampled object depth centroids
in a common camera frame. When the 90th-percentile displacement exceeds both 50 mm and
25% of the inferred long dimension, it refuses the stationary fit and asks for review.
On the full moving-box video the sampled displacement was 110 mm and the builder refused;
on the short stationary test it was 17.5 mm. This is only a conservative heuristic:
changing viewpoints, partial masks, and camera tracking error can also move visible-depth
centroids. It does not prove stationarity and needs validation on moving-camera captures.

The stationary builder can now create an upright cylinder as well as a box when the agent
selects that visible shape. Cylinder diameter is a circular-shape hypothesis based on the
larger observed horizontal extent, not a measurement of hidden sides. Synthetic geometry
tests check the circular profile and that MuJoCo can load the resulting mesh. No real
cylinder video has yet been processed, so classification, scale, replay, and generalisation
for this branch remain unverified.

A synthetic RGB-D test with a known 120 mm diameter, 200 mm tall upright cylinder exposed
the previous diameter heuristic: averaging silhouette width with the shallow visible-depth
extent gave about 66 mm diameter and 56.5% replay mask coverage. Using the larger horizontal
extent gives about 103 mm diameter and 90.6% coverage on the same 16 synthetic frames;
height was estimated at 187 mm. The test used generated masks and a fixed camera and does
not establish performance on real video, hidden surfaces, or arbitrary cylindrical shapes.

Three further ideal synthetic cylinders (true diameters 80, 180, and 120 mm at different
distances) exposed 18–20% diameter underestimates with the larger-extent heuristic. A
new guarded circle fit estimates the center and radius from the observed depth arc and
falls back to the extent when curvature support is insufficient. On these noiseless
synthetic scenes, generated diameters were 80, 180, and 120 mm and mask coverages were
93.3%, 93.6%, and 92.8%. Observed envelope sizes are still stored separately from generated
sizes. These ideal fixed-camera scenes have no real sensor noise or segmentation errors,
so the results do not validate real-world dimensions or shape classification.

For a 120 mm synthetic cylinder with independent per-pixel depth noise (2 mm and 5 mm
standard deviation, and a case with 2 mm noise plus 5% large outliers), the initial
algebraic curve fit estimated diameters of 112.3, 102.0 and 111.9 mm, with mask
coverage 89.8%, 80.6% and 90.1%. A robust geometric-distance refinement estimated
118.5, 118.6 and 117.9 mm, with coverage 93.5%, 94.0% and 93.9%, respectively.
The masks, fixed view and simple cylinder are synthetic; these numbers are not a real
sensor accuracy claim or evidence of generalisation to arbitrary round objects.

A negative trial deliberately fitted a cylinder to the real stationary cardboard-box
clip. Curvature fitting failed and the fallback cylinder still covered 81.2% of the mask,
showing that a coverage score alone cannot identify the correct shape. The builder now
warns when a cylinder has no supported circular fit, and appearance assessment warns when
video classification says box while the chosen static shape is a cylinder. The fallback
still permits a cylinder when curvature cannot be established because partial observations
can be ambiguous; these warnings are not proof that an incorrect shape will always be caught.
The generated-size metadata was also corrected to report the cylinder's actual equal
horizontal diameters instead of the unequal observed box extents.

The standalone appearance command now accepts `--simple`, matching the unified command's
simple-colour default. Reusing the saved classification on a copy of the static-box result
with this option completed locally. This is a resumed run and does not test classification
or the full automated pipeline a second time.

Static appearance processing now records sampled silhouette IoU as well as coverage and
shows the value in the viewer when available. On the 24-frame real box clip, the box
visual's sampled IoU was 0.737 and the deliberately wrong cylinder's was 0.686; their
coverages were 93.6% and 81.5%, respectively. These use the same automatically tracked
mask, not independent annotations, and a higher score alone cannot establish object
identity or correctness of unseen surfaces. The static comparison renders the retained
model twice and is not presented as an improvement over a separate baseline.

A synthetic moving-camera test translated the camera 100 mm laterally around a stationary
120 mm diameter cylinder, with known camera poses and perfect masks/depth. The model's
estimated diameter was 120 mm, the replay mask coverage was 93.3%, and the sampled
centroid-shift estimate was 4.7 mm. A reproducible integration test now exercises this
coordinate/replay path. It does not model real odometry drift, rolling shutter, occlusion,
shape classification, or sensor noise and therefore does not validate real phone captures.

A texture-visibility bug let two faces of the same mesh share a visibility result: a
hidden rear face projected onto its own visible front face and was reported as observed.
On the short fixed-view box, the old selection reported all 6 faces as video-observed.
The selector now checks rendered surface depth at each projected sample, with a 15 mm
minimum tolerance for downsampled depth; the same box reports 2 observed and 4 inferred
faces. This tolerance and the mask do not prove every sampled pixel is unoccluded.
A separate existing laptop fit completed texture processing with the new depth check
(base 4/14 face groups, lid visuals 6/6 and 2/6); these are processing results, not a
validation of the laptop texture's appearance or a fresh run.

On a copy of the independent laptop fit, depth-aware texturing still put a trackpad image
on the outer lid: the lid track had drifted onto the base on several open frames. Texture
selection now excludes individual lid masks mostly overlapping the base when an open
indicator is visible. In a repeat of this existing fit, the outer lid render showed the
Apple logo rather than the trackpad, while the front retained a recognizable screen and
keyboard. Some edge bands remain imperfect; the open indicator and masks can themselves
be wrong. This was visual inspection of one reused fit, not fresh autonomous validation.

The laptop's remaining rear-lid band was traced to a mixed closed-lid mask containing
part of the base when the open indicator was absent. Texture selection now excludes base
pixels from lid candidates when the indicator is visible or the plate track has become
nearly identical to the base. A repeated render of the same existing fit showed a clean
logo-bearing rear lid and retained the screen and keyboard on the front. This is still a
heuristic based on tracked masks; incorrect base masks or other viewpoints can cause
missing or wrong texture, and this is not a new end-to-end success.

Rechecked against the original `data/repeat/run{1,2,3}.log` and `model3d.json`: only
1/3 original runs reached the 70% tracked-mask coverage threshold for both lid and base
(75/86%, 93/16%, 93/31%). Their recorded lid widths were 316.3, 304.5 and 304.3 mm
(range 12.0 mm), with depths 216.2, 215.1 and 213.6 mm (range 2.6 mm). All had at
least one recorded tracking or thickness limitation, so this is a mask-coverage count,
not a count of reliable end-to-end successes. These runs predate later changes and all
use one capture; they do not measure the success rate of the current pipeline.

An independent ordinary-video run processed the same short static-box clip through
MapAnything, without a reference or scale correction, and then used the open-weight agent
and static box pipeline. It classified category `box`, generated a model and viewer, and
covered 98.4% of its own tracked mask over 24 frames. Its observed envelope was about
370 × 291 × 537 mm, versus 131 × 112 × 214 mm from a separate LiDAR-based fit of the
same clip. The two fits use different estimated poses and masks and the LiDAR values are
not independent ground truth, but the large difference demonstrates that high mask coverage
does not establish RGB metric scale. The report from this run used misleading measurement
language. The agent now receives an explicit note when its depth comes from MapAnything,
so future reports are instructed to call dimensions estimates; this instruction was added
after the run and its effect has not been verified in a fresh agent execution.

A manual calibration experiment used two interior points on the box's visible side in
frame 8. Their distance was estimated from the paired LiDAR depth as 148.1 mm; the RGB
capture estimated 397.7 mm before scaling, yielding scale ×0.3723. A different pair on
the same face was 52.9 mm by the LiDAR depth and 47.0 mm after RGB scaling (−11.2%).
Refitting the existing RGB tracks after scaling produced an observed envelope of about
140 × 108 × 208 mm versus the separate LiDAR fit's 131 × 112 × 214 mm. Masks, orientations
and sources differ; the reference itself is a LiDAR estimate, not a ruler measurement, and
the second pair on the same face is not a broad independent validation. The calibrated
capture and fit are copies created for the experiment, not a new automatic run.

The new calibrated-capture command was exercised through a fresh agent run on the same
ordinary-video box clip. The agent reused the calibrated depth, fitted a 134 × 106 × 209 mm
static envelope, and completed export. Its separate classification call identified the
table rather than the tracked object. Static classification now receives crops from the
tracked object's bounding boxes alongside the full video and is instructed to identify the
target, not the supporting surface. Rerunning only classification and appearance on a copy
of that result returned `box` and paper/cardboard evidence. This is a resumed classification
on one existing clip, not a new full autonomous success or a guarantee that a bad tracked
mask cannot lead to a wrong object. The calibration reference is the paired LiDAR estimate,
not an independent physical measurement; the second check differs by −11.2%.

A second fresh agent run reused the calibrated RGB capture and used the object-crop
classification prompt from the outset. It segmented and tracked the box and table,
fitted a 133 × 107 × 208 mm box, classified it as an open white cardboard carton,
and exported a viewer. It initially passed the track name `box` to physics and failed;
the agent retried with model link `base` and succeeded. The code now maps the static
track name to `base` for future calls, but the running process predated this fix.
The final report acknowledged estimated scale but still called one replay a
"depth-measured model"; the user-facing source and quality warnings remain necessary.
This is a single 24-frame repeat of the same source video with a LiDAR-derived reference,
not a new object, independent metric ground truth, or a general success rate.

The viewer now labels models using MapAnything depth as estimated RGB depth, and new
replay overlays say "depth-based model" rather than "depth-measured model". Existing
rendered replay videos retain their original burned-in wording until regenerated.

Classification crops now include tracked parts of articulated objects as well as static
objects. A resumed appearance run on copies of existing LiDAR fits classified the laptop
as `MacBook laptop` and the cabinet as `three-drawer cabinet` with handles. Both accepted
their visual candidates. The drawer's sampled mask coverage was 62% for its base and 63%
for its moving drawer, so the fit still needs review. These are two existing videos and
existing tracks, not fresh end-to-end runs or evidence of a classification success rate.

The lid's filtered observation mask is now reused during atlas painting as well as
frame selection. On a copy of the existing laptop fit, video texturing completed;
an open-pose render shows a keyboard and trackpad on the base and the logo on the
lid's outside. A visible hinge gap and edge artifacts remain at that pose. This is
one reused fit, not a new end-to-end run or proof that all mis-tracked pixels are excluded.
For this fit, the lid joint axis is at the base body's origin; the fitted base bounds
end at z = -12.8 mm in that body's coordinates, with the closest y bound at +2.6 mm.
The straightened base visual has the same bounds. This is consistent with the apparent
gap in the open-pose render, but does not establish the distance between complete mesh
surfaces at every angle. Moving the joint or changing collisions would also change the
fitted motion, so a visual-only connector still needs a separately validated design.
An exploratory copy adds a non-colliding, massless cylinder near that axis. At a single
open pose a 14 mm radius looked too bulky; an 8 mm radius with a 6 mm vertical offset
looked less intrusive and partially concealed the gap. Neither placement has been
validated across the hinge range or against the video. The experiment is MJCF-only and
has not been integrated into the standard generator or URDF export.
The smaller connector was rendered at three sampled joint positions (approximately
-103°, -23° and +11° in the fitted joint coordinates). The model remained recognizable;
there were still edge and spacing artifacts. On the same video's sampled masks, base
coverage changed from 91.12% to 91.48% and base IoU from 0.567 to 0.556; lid coverage
changed from 75.66% to 75.09% and lid IoU from 0.535 to 0.536. These comparisons use
tracked masks, not independent truth. The connector dimensions were hand-selected for
this one model, so they do not yet establish a reusable generation rule.

A later implementation computes a small connector from the fitted hinge and base bounds
for compatible x-axis laptop hinges, includes it in the existing sampled visual guard,
and emits it in MJCF and URDF as visual-only geometry. A copy of a second existing fit
of the same laptop passed that guard (base IoU 0.562 to 0.557, lid IoU 0.735 to 0.733)
and looked laptop-like at three sampled joint positions. This is not a new object or
an end-to-end evaluation; models with incompatible or distant hinges skip the connector.

A fresh open-weight agent run on the existing scaled MapAnything laptop capture retracked
the base and lid. Its second fit reported base/lid mask coverage of 75%/82%, versus
54%/61% initially, but an estimated 68 mm base thickness and a 298 x 136 mm lid. The
provider then returned HTTP 402 before the agent completed. On a copy of that incomplete
fit, appearance was resumed using classification from a separate run on the same video;
the straightened visual passed the sampled mask comparison. In three sampled poses, the
open model looks like a simple laptop but the near-closed pose has a lid underneath the
base and the body appears raised above the floor. This is not a successful autonomous
end-to-end result, and physics/collisions do not follow the heuristic visual thickness.
The source video shows the laptop closed at frames 461, 500, and 600, whereas fitted
angles there are approximately -141°, -120°, and -129°. Clamping a copy's replay to the
estimated -118° closed angle still produces a visibly misplaced lid. At -110° the lid is
approximately flat but overhangs the fitted base. This indicates that angle clipping by
itself is not a validated correction; the underlying fit/visual alignment also needs work.
An exploratory copy aligns the visual base edge to the hinge, constructs visual lid and
screen panels from that base footprint in a manually selected -110° closed pose, and
clamps its replay angle at that value. Three renders look like a closed or open laptop.
On sampled tracked masks, lid coverage/IoU changed from 0.649/0.532 to 0.833/0.651;
base coverage changed from 0.871 to 0.879 (base IoU is not informative for its partial
mask). These numbers compare the two different visuals and trajectories, not independent
ground truth. This was a manually tuned experiment on one fit; fitted collision, joint
limits, and pre-existing replay video were not regenerated and therefore do not match
the modified appearance/trajectory. It is not a deployable or end-to-end result.
An exploratory rule estimated the closed angle directly from the lid's long in-plane
direction, obtaining -113.6° on the RGB fit and -105.2° on a separate LiDAR fit of the
same laptop. With corresponding aligned visuals and clamped replay angles, the RGB fit's
sampled lid IoU was 0.656 versus 0.532 before; the LiDAR fit's was 0.703 versus 0.733.
The sampled masks are not independent truth, and the LiDAR comparison has only 10 lid
frames. As above, these copies contain manually staged visual and replay changes with
stale collision/limits/video; they are experiments, not consistent exported models.
The calculation is reproducible through `scripts/experimental_laptop_align.py SOURCE DEST`.
It creates a separate review-only copy, retains the original trajectory, and removes stale
replay media; it does not repair collision geometry, inertia, or joint limits. On the RGB
fit it selected -113.6° and clamped 275 frames. Three rendered poses show a recognizable
laptop, but the model remains visibly elevated above the floor and is not physics-ready.
The experimental replay was subsequently regenerated over all 923 source frames, with
adjusted angles explicitly labelled. Its tracked-mask coverage averaged 88.6% for the
base on 523 frames and 84.8% for the lid on 923 frames; these are not ground truth, and
the lid track has known identity switches. Inspection of six frames showed reasonable
coarse overlap in several views but clear offset in the partly open pose at frame 700.
The estimated closed-frame geometry and unchanged physics remain unverified.
Another manually modified copy uses the aligned base/lid meshes for collisions, updates
the lower hinge limit to the experimental closed angle, and translates both model and
recorded camera poses so the new base rests on the floor. Its three rendered poses look
like an open/closed laptop without the earlier floating base. The sampled lid IoU remained
0.656 after the common coordinate shift. MuJoCo loaded both MJCF and URDF; inferred MJCF
body masses were 0.79 kg and 0.26 kg from default density, not measured masses. This is
still a manual, single-object experiment: stale provenance and replay media remain in the
copy, and motion, inertia, contact and simulation behaviour have not been validated.
The optional `--coherent-collision` argument now reproduces that geometry, hinge-limit,
and common camera/model shift on a new copy. It rejects source models with assigned
physics or explicit inertials, and records that collision and physics are hypotheses.
On the same RGB fit both generated MJCF and URDF loaded; this does not establish physical
validity or performance on another object.
In a three-second free-dynamics check of the generated MJCF, joint angles stayed finite,
but MuJoCo's soft limits allowed about 4.9° of travel past the lower limit and 2.5° past
the upper limit in two high-initial-velocity trials. The model excludes base-lid contact;
the aligned visual is therefore not a guarantee against visible intersection during
dynamic simulation. Masses and this test's initial velocities were not measured.
On a separate copy, reducing the MuJoCo limit solver time constant from its default to
4 ms reduced those two peak overruns to about 0.84° and 0.42°. The coherent-collision
experiment now sets that time constant to at least twice the configured timestep; this
is solver tuning, not measured hinge behaviour, and URDF engines may behave differently.

A manually seeded 40-frame closed segment of the real drawer2 LiDAR video was fitted as a
stationary box. Its fitted envelope was 374 x 12 x 481 mm and the original replay covered
67.5% of its tracked mask. For a manually supplied three-drawer classification, a visual
cabinet depth of 243 mm (65% of visible width) and three fronts were hypothesized. The
generated visual passed the sampled mask guard (81.9% coverage); a front render looks like
a three-drawer cabinet. This depth is not measured, the collision remains the thin fitted
envelope, and the segmentation prompts and classification were manually supplied. The
40 frames are one clip from an existing video, not an independent autonomous success.
Rerunning classification on the existing manually selected tracks identified a three-drawer
cabinet with handles and accepted the same visual. This exercises automatic classification
and completion after a manual fit; it does not verify autonomous segmentation or tracking.
An additional fresh Spark + open-SAM run on that same 40-frame RGB-D clip automatically
segmented and tracked the cabinet and floor, fitted the static envelope, classified a
three-drawer cabinet with handles, and completed export. The hypothetical cabinet visual
passed the sampled comparison (base mask coverage 67.5% to 81.9%); a browser render
shows three fronts. The visible-depth envelope remains approximately 374 x 12 x 481 mm;
the 243 mm visual depth is assumed, while collision remains thin. The front-direction
metadata in this already-completed run was added separately for the viewer after the run.
This is one short segment of the existing video, not a general success rate or a new object.
The same 40-frame clip was also run from ordinary RGB video through MapAnything, Spark,
open SAM, fit, classification and viewer export. It correctly classified a three-drawer
cabinet with handles, but the estimated envelope was approximately 2578 x 2449 x 150 mm
and base replay coverage was 60.3%. The exported browser model looks like a flat slab,
not a cabinet. The scale was not calibrated and its upright/depth orientation is unreliable;
this is an end-to-end failure for the requested recognizable shape, not a success.
In a separate manual investigation, open SAM detected three handles inside the tracked
cabinet on frame 20 (and a fourth handle outside it). The three RGB-depth handle centres
indicated an up direction far from MapAnything's estimated up. Supplying that direction
and a manually chosen sign to a fresh fit produced an upright visual that looks like a
three-drawer cabinet; its uncalibrated visible envelope was 2587 x 115 x 2396 mm and its
hidden visual depth was assumed. The upright choice, handle filtering and classification
reuse were manually staged; this does not demonstrate automatic correction or reliable
metric dimensions. The initial fitted mask coverage was 58%, so review remains necessary.
A subsequent fresh end-to-end run on the same RGB capture automatically requested the
handle-based upright hint, detected aligned handles on frame 7, compared against the
original orientation, and restored the handle-based fit. Classification identified a
three-drawer cabinet and the exported viewer shows an upright cabinet with three fronts.
Sampled mask coverage increased from 56.5% for the fitted box to 70.0% for the inferred
visual; these are comparisons to the tracked mask, not independent ground truth. The
uncalibrated observed envelope was 2586 x 139 x 2394 mm and the assumed visual depth
was 1681 mm, so neither is a reliable real-world measurement. This is a repeat on the
same short clip, not evidence of general success on other objects or videos. The collision
remains the thin fitted box, and the replay displays the earlier fit, not the completed
visual. Output: `data/appearance_trials/static_drawer_rgb_auto_up_viewer`.
`scripts/experimental_drawer_collision.py SOURCE DEST` can create a separate review-only
copy using the hypothetical visual envelope for collision and recomputing uniform-density
inertia at the original estimated mass. For the one drawer run this yielded 374 x 243 x
481 mm collision bounds and retained 1.2 kg; the implied solid density is only 27 kg/m³.
The original 12 mm fit and physics record remain preserved. This does not measure hidden
depth or validate mass, friction, contact, or robotic interaction.

`scripts/video_to_model.py` now attempts a separate laptop review candidate when a laptop
fit reports over-closing. The original remains untouched; the candidate adjusts the visual
and replay angles but retains original collision, inertia and joint limits. When compatible,
an experimental overlay is rendered and the viewer includes both results. The viewer limits
its candidate slider to the hypothesized closed pose, but exported physics remains unverified.
Unsupported meshes or existing candidate directories are left alone. This trigger and
alignment are heuristics, not a guarantee that tracking or unobserved geometry is correct.
The candidate and original are now additionally rendered with their respective joint
trajectories against the same sampled tracked masks. On the existing RGB laptop run,
35 base samples showed coverage 87.1% to 88.2%, and 60 lid samples showed 64.9% to
84.4%; no sampled regression was detected. Base IoU was not treated as informative
because its tracked mask covers a partial component. These masks can drift or switch
surfaces, so this comparison is not independent validation of the geometry or motion.

An additional autonomous RGB run used the separate `notebook_2_mapanything` video.
The agent recognized a laptop and exported a straight-edged model after several tracking
attempts. The final fitted replay covered 46% of the lid track and 83% of the base track;
the base was scored on only 142 of 408 frames. The viewer shows a recognizable keyboard
base but an oversized, visually disproportionate lid. The footage partially crops the
near laptop, includes another laptop, and has a hand occluding the moving lid. This is
not a successful end-to-end example of the requested coherent shape. Output and warnings:
`data/appearance_trials/notebook2_rgb_fresh_viewer`. No independent metric scale was verified.
An offline visual-only trial shortened this lid to 75% of its fitted width and inset the
screen to create a bezel. On 60 identical sampled tracked masks, lid coverage decreased
from 59.8% to 47.2% (IoU 0.423 to 0.379). The candidate failed the coverage guard and
was not activated. This mask comparison is not ground truth, but it gives no support for
silently replacing the video-derived fit with that fixed proportion. The trial copy is
labelled `data/appearance_trials/notebook2_proportion_experiment/experimental_proportions.json`;
its original replay and physics do not represent the modified visual meshes.
Separately, extending the cropped base visual to the aligned lid's width while retaining
the lid dimensions and adding a bezel gave 87.4% sampled base coverage versus 84.2%
before; lid coverage changed from 59.8% to 58.6%, within the comparison threshold.
The method is now limited to small aligned width differences and applies to visuals only;
the normal sampled-mask guard still applies. The trial used the same tracked video, not
independent measurements, and has not been tested as a fresh end-to-end run.
To exercise the implementation rather than only the manual mesh trial, a copy of this
fit had its inferred visuals removed by hand, then appearance was resumed using the
earlier classification. The completion code set both `screen_bezel_assumed` and
`visual_alignment_assumed`; its sampled guard retained the result (base coverage
84.0% to 87.4%, lid 47.7% to 58.6% compared with the fitted visual). This is still
the same video and a manually prepared fit, not an uninterrupted end-to-end test.
The completion code also limits visually extreme portrait-like laptop lids by at most
15%, with the same mask guard and an explicit assumption warning. In the same manually
prepared fit, it selected a 0.856 height factor; sampled lid coverage was 47.7% for the
fitted visual and 54.9% for the completed one. This does not validate the true lid size,
motion, or collision and does not replace a fresh-video test.

An older planar fit of `data/real/stand.mov` exports a recognizable three-piece monitor
stand with two joints (`data/appearance_trials/stand_existing_viewer`). This fit predates
the current appearance workflow and does not establish automatic object classification
or RGB-depth reconstruction. The current `video_to_model.py` path requires a 3D fit and
does not automatically select the multi-joint planar path. An explicit `--planar` option
now runs the planar agent and object classification from an ordinary side-view video,
then exports the viewer. Depth and potentially scale are assumed, and SAM 3D and video
texture options are not supported on that path. On a copy of the existing stand fit,
an actual classification call labelled the monitor stand an "adjustable desk lamp".
This is a classification failure; the geometry was not regenerated and this is not a
fresh end-to-end validation. Broader support for other articulated objects remains unproven.
After asking the classifier to avoid asserting a function without visible evidence,
one repeat on that same existing fit returned "articulated stand". That broader label
avoids the unsupported lamp claim but does not identify the precise use. It is one
repeat on the same video, not a general classification accuracy measurement.
A fresh `--planar` run on that video automatically segmented and tracked the base, arm,
and top through 462 frames, fitted a two-hinge model, classified it as "articulated
stand", and exported `data/appearance_trials/stand_planar_fresh_viewer`. The model looks
like a stand, but its base is bulky and irregular, and the mean arm replay IoU is 0.56;
therefore this is not yet a clean straight-edged result. The scale (3200 px/m) and
cross-view widths were assumed. The second joint moved only about seven degrees in the
fit, so its true motion is not established. This is one run on an existing video, not a
success rate or proof of general support.
An offline stand trial replaced the base visual with a rectangle: reference-frame base
mask IoU changed from 0.799 to 0.779. Reducing the base's apparent thickness to 8% of
its visible width then reduced reference-frame IoU to 0.290. The original thick silhouette
may include perspective from the base's top surface, and one side-view frame cannot
establish its true thickness. This candidate was not activated; its collision and replay
remain the old fit. Rectangularizing the other two links also substantially reduced their
reference-frame mask agreement. No full-video guard has been implemented for planar shape
completion yet.
For the existing run, root motion was reconstructed from the original masks and video
using the fit's pose algorithm. A new planar comparison renders both models at 60 shared
frames with that motion: baseline base mask IoU 0.808, full-thickness rectangular visual
0.734, thin rectangular visual 0.259. These sampled tracked-mask comparisons are not
ground truth. Future planar fits save root motion to `planar_replay.npz` for the same check;
the comparison is available but does not automatically change or approve a model.
The planar appearance step now tries a convex, straight-sided base visual for an object
classified as a stand, and accepts it only when at least 10 sampled masks show no more
than five percentage points of IoU loss. It preserves the fitted collision mesh. On a
copy of the existing stand fit with reconstructed root poses and reused classification,
the actual completion code accepted the candidate (0.808 to 0.783 mean sampled IoU).
The visible thickness remained unchanged and remains an assumption; this is not a new
uninterrupted run or proof of performance on other stands.
The RGB-D agent can now choose the planar fitter for a side-view chain with at least
two explicit parent links, even if a depth capture was prepared. This route records its
own planar provenance: depth and scale are assumed rather than attributed to the depth
capture. A single hinge and drawers continue to use their depth-specific fitters.
`--planar` remains available to select the planar route explicitly.
An independent automatic run on the same stand video (no `--planar` flag) prepared a
MapAnything capture, identified a two-joint chain and chose `fit` itself. After re-anchoring
the base, fitted mean replay IoUs were 0.83 base, 0.57 lower arm and 0.82 top; it classified
the object as an "articulated stand" and activated the convex base visual (sampled base
IoU 0.835 to 0.807). A browser render looks like a stand but the base remains thick and
the arm tracking needs review. The planar result correctly records assumed depth/scale;
no physics was assigned. The running process started before per-part IoU persistence was
added, so its quality file notes scores were not recorded, though the agent report gives
the fit summaries above. This is one existing video, not a broad success rate.
On a separate existing `stand_2.MOV` recording with a white base and two long slim arms,
the same classifier returned "articulated dual-arm stand". A browser render of its older
planar fit shows the approximate base and arms, but the video's precise object use was not
independently established here. This was classification of a pre-existing fit, not a new
end-to-end run, and the older fit lacks the recorded root motion required for the new
planar shape comparison.
