# Demo narration

Total 171 s. Read at a calm pace; each line starts at its time.

**0:00.0  Video2URDF**  
Robots need simulation models of the objects they handle, and people build them by hand. Video2URDF measures them from one phone video, and a robot can use them right away.

**0:12.5  1  Capture**  
I film the object with an iPhone while I move it. No markers and no calibration card: the LiDAR gives metric depth. A plain video works too.

**0:23.9  2  Agent**  
Muse Spark runs the pipeline as an agent. It watches the video, chooses noun prompts for SAM 3.1, picks the right fitter, and judges what it sees: for this desk lamp it decided the pole never moves.

**0:39.2  3  Measure and verify**  
SAM 3.1 tracks every part through all nine hundred frames, and MuJoCo replays the measured model from the phone's own camera: it covers ninety five percent of the lid. The lid measures three oh nine by two oh five millimetres; the spec says three oh four by two twelve.

**0:59.0  4  It checks itself**  
Spark checks its own work. Its first model had the base as a thin sliver. Spark watched the replay, segmented the base again, and the base came back.

**1:10.8  5  It knows what it is**  
The measured meshes have the right sizes but look like blobs. So Spark names the object and rebuilds it cleanly: a cabinet with fronts and handles, a lamp with its LED bar. Each round three Meta models propose a design, and the one that fits the video best wins.

**1:30.6  6  It looks real**  
For a curved object, SAM 3D Objects builds a realistic mesh, placed by our LiDAR depth: this mouse lies half a millimetre from its measured surface. Circling it with the phone gives its size from every side.

**1:45.8  7  The robot uses it**  
In MuJoCo physics the Panda grips the handle Spark drew and pulls the drawer out, then carries the mouse and turns it. Only the grip moves them.

**1:57.2  8  Physics**  
Spark judges materials and masses. Where a part stands with no hand on it, the joint friction is measured, and checked by letting go in MuJoCo.

**2:08.2  9  Benchmark**  
On the RBO dataset, real RGB-D videos with motion capture, the median hinge axis error over thirteen sequences is two and a half degrees.

**2:18.5  10  URDF Doctor**  
URDF Doctor checks the models you already have against a video. Three injected bugs and a sign flip in a joint log: all found.

**2:28.7  11  Open weights**  
Every model it calls is a Meta model, and it runs end to end on open weights: Muse Spark Open with SAM weights, or Muse Glimmer on our own GPU.

**2:41.2  Try it**  
Everything is in the browser, with URDF downloads and every step the agent took. One video in, a model a robot can use out.
