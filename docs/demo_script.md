# Demo narration

Total 177 s. Paced to a synthesized voice: each line starts at its time, after about 0.3 s of silence.

**0:00.0  Video2URDF**  
Robots need simulation models of the objects they handle, and people build them by hand. Video2URDF measures them from one phone video, and a robot can use them right away.

**0:12.1  1  Capture**  
I film the object with an iPhone while I move it: no markers, no card. The LiDAR gives metric depth.

**0:18.9  2  Agent**  
Muse Spark runs the pipeline as an agent. It watches the video, prompts SAM 3.1 with nouns, picks the fitter, and judges what it sees: for this desk lamp, it decided the pole never moves.

**0:32.2  3  Measure and verify**  
MuJoCo replays the model from the phone's camera and covers ninety five percent of the lid: three oh nine by two oh five millimetres, against a spec of three oh four by two twelve.

**0:44.0  4  Why not just ask an AI?**  
Why not just ask an AI? Asked plainly, the same Spark recalls a MacBook's spec, but in eight of nineteen answers its hinge can't open as far as the video shows, and it gives every drawer the same travel. With tools, it measures them.

**0:59.3  5  It checks itself**  
Spark checks its own work. On a microwave it had never seen, its first model was wrong: a door twice too big and no body. It watched the replay, tracked the floor and the body a second time, and refit: now the door swings on a vertical hinge, where the real one is.

**1:16.2  6  It builds what it never saw**  
This microwave sits in a shelf; its back was never filmed. The front is measured; the video bounds the depth to three thirty four to three eighty seven millimetres. Muse Spark recognises a Samsung twenty three litre and puts it at three sixty five, labelled an estimate. A probe pushed at the back stops on it.

**1:35.7  7  It knows what it is**  
Measured meshes have the right sizes but look like blobs, so Spark names the object and rebuilds it from clean parts: three Meta models propose, and the video picks.

**1:46.5  8  It looks real**  
For a curved object, SAM 3D Objects builds a realistic mesh, placed by our LiDAR depth: the phone's depth points lie half a millimetre from it, at the median.

**1:57.0  9  The robot uses it**  
In MuJoCo, a reinforcement learning policy trained on the measured drawer pinches its handle and pulls it open, nineteen times in twenty on the full arm. The Panda also opens and shuts the microwave door and carries the mouse. Spark judges masses; joint friction is measured where a part stands alone.

**2:16.1  10  Benchmark**  
On the RBO motion-capture benchmark, laptops and books get a median hinge axis error under two degrees, under three on open weights; box lids still fail.

**2:27.0  11  URDF Doctor**  
URDF Doctor checks the models you already have against a video. Three injected bugs and a joint log with its sign flipped: all found.

**2:36.6  12  Open weights**  
Every model it calls is a Meta model, and it runs end to end on open weights, Muse Spark Open or Muse Glimmer on our own GPU, reproduced from a fresh clone.

**2:47.4  Try it**  
Everything is in the browser, with URDF downloads and every step the agent took. One video in, a model a robot can use out.
