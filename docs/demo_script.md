# Demo narration

Total 171 s. Read at a calm pace; each line starts at its time.

**0:00.0  Video2URDF**  
Robots need simulation models of the objects they handle, and people build them by hand. Video2URDF measures them from one phone video, and a robot can use them right away.

**0:12.5  1  Capture**  
I film the object with an iPhone while I move it: no markers, no card. The LiDAR gives metric depth.

**0:21.2  2  Agent**  
Muse Spark runs the pipeline as an agent. It watches the video, chooses noun prompts for SAM 3.1, picks the right fitter, and judges what it sees: for this desk lamp it decided the pole never moves.

**0:36.5  3  Measure and verify**  
MuJoCo replays the measured model from the phone's own camera; it covers ninety five percent of the lid, which measures three oh nine by two oh five millimetres against a spec of three oh four by two twelve.

**0:52.1  4  It checks itself**  
Spark can't finish over its own warnings. Its first finish here was refused: the screen's track had slid onto the lid's back. It re-tracked the lid and reported what it couldn't verify. Five fresh runs land within four millimetres of the spec.

**1:09.2  5  It knows what it is**  
The measured meshes have the right sizes but look like blobs, so Spark names the object and rebuilds it from clean parts. Each round three Meta models propose a design, and the video picks the best.

**1:24.1  6  It looks real**  
For a curved object, SAM 3D Objects builds a realistic mesh, placed by our LiDAR depth: the depth the phone measured lies within half a millimetre of the mouse's mesh. Circling it with the phone gives its size from every side.

**1:40.8  7  The robot uses it**  
In MuJoCo physics the Panda grips the handle Spark drew and pulls the drawer out of its measured cabinet, then carries the mouse and turns it. Only the grip moves them.

**1:53.8  8  Physics**  
Spark judges materials and masses. Where a part stands with no hand on it, the joint friction is measured, and checked by letting go in MuJoCo.

**2:04.8  9  Benchmark**  
On the RBO motion-capture benchmark, laptops and books come out with a median hinge axis error of two degrees, under three on open weights alone; box lids still fail.

**2:16.9  10  URDF Doctor**  
URDF Doctor checks the models you already have against a video. Three injected bugs and a sign flip in a joint log: all found.

**2:27.2  11  Open weights**  
Every model it calls is a Meta model, and it runs end to end on open weights, Muse Spark Open or Muse Glimmer on our own GPU, reproduced from a fresh clone.

**2:40.5  Try it**  
Everything is in the browser, with URDF downloads and every step the agent took. One video in, a model a robot can use out.
