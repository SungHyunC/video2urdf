# Demo narration

Total 168 s. Read at a calm pace; each line starts at its time.

**0:00.0  Video2URDF**  
Robots need simulation models of the objects they handle, and people build them by hand. Video2URDF measures them from one phone video. This laptop was filmed once; the robot is closing the model measured from that video.

**0:15.2  1  Capture**  
I film my laptop with an iPhone while I open and close it. No markers and no calibration card: the LiDAR gives metric depth.

**0:25.5  2  Agent**  
Muse Spark runs the pipeline as an agent. It watches the whole video, chooses short noun prompts for SAM 3.1, and rephrases when a prompt finds nothing.

**0:36.8  3  Measure and verify**  
SAM 3.1 tracks every part through all nine hundred frames. MuJoCo then replays the measured model from the phone's own camera, on every frame: it covers ninety five percent of the lid. The lid measures three oh nine by two oh five millimetres; the spec says three oh four by two twelve.

**0:57.8  4  It checks itself**  
Spark checks its own work. In this run its first model had the base as a thin sliver. Spark watched the replay, segmented the base again and re-tracked, and the base came back. The fit tool also names such mistakes in plain words.

**1:15.4  5  It looks real**  
The measured shapes are painted with the video itself. Every face takes the frame where it was seen best, so the screen shows its wallpaper, the base its keyboard, and the closed lid its logo.

**1:29.8  6  Physics**  
A robot needs mass and friction too. Spark judges the materials and masses. Where the lid stands with no hand on it, only hinge friction holds it, so the friction is measured there. Let go in MuJoCo, the lid stays; without that friction it falls.

**1:48.2  7  Drawers too**  
Drawers work too. The agent finds which drawer moves and measures two hundred and forty eight millimetres of travel against the cabinet.

**1:57.6  8  Benchmark**  
On the RBO dataset, real RGB-D videos with motion capture, the agent runs every sequence from scratch. Over thirteen sequences the median hinge axis error is two and a half degrees. Box lids still fail.

**2:12.1  9  URDF Doctor**  
URDF Doctor checks the models you already have against the video. Three injected bugs and a sign flip in a joint log: all found.

**2:22.3  10  Open weights**  
Every model it calls is a Meta model, and it runs end to end on open weights: Muse Glimmer on our own GPU, with SAM 3.1 weights, through to the physics.

**2:35.2  Try it**  
Everything is in the browser: the video, the replay, the robot and the painted model, with URDF downloads. One video in. A model that moves like the real thing out.
