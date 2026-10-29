# Demo narration

Total 157 s. Read at a calm pace; each line starts at its time.

**0:00.0  Video2URDF**  
Robots need simulation models of the objects they handle. Today, people build them by hand. Video2URDF measures them from one phone video.

**0:09.5  1  Capture**  
I film my laptop with an iPhone while I open and close it. No markers and no calibration card: the LiDAR gives metric depth.

**0:19.7  2  Agent**  
Muse Spark runs the pipeline as an agent. It looks at frames, and chooses short noun prompts for SAM 3.1. When a prompt finds nothing, it rephrases. Parts that disappear, like the screen when the lid closes, get anchors on several frames.

**0:36.8  3  Measure**  
SAM 3.1 tracks every part through all nine hundred frames. From the depth of the lid in each frame, the hinge is fit by least squares, where the lid and the base planes meet, and the opening is read frame by frame.

**0:54.0  4  Verify**  
Then the check. MuJoCo replays the measured model from the phone's own camera pose, on every frame, on top of the video. It covers ninety five percent of the lid and ninety four percent of the base. The lid measures three hundred and nine by two hundred and five millimetres. The spec sheet says three oh four by two twelve.

**1:18.1  5  Use**  
The URDF is ready to use. A Franka arm in MuJoCo closes the measured laptop by pushing along the measured hinge.

**1:27.2  6  Benchmark**  
Does it generalise? On the RBO dataset, real RGB-D videos with motion capture, the agent runs every sequence from scratch. Across thirteen sequences the median hinge axis error is two and a half degrees, and the angle error three point three degrees, even when people push the whole object around. Box lids are where it still fails.

**1:50.1  7  URDF Doctor**  
It also checks the models you already have. URDF Doctor compares a URDF with the video. One written from the published specs passes. One with three injected bugs: a moved hinge, a tilted axis, and a joint limit that is too tight. All three are found, and so is a sign flip in a joint log.

**2:12.6  8  Open weights**  
Every model it calls is a Meta model, and one flag runs the whole pipeline on open weights: Muse Spark Open, and SAM weights on our own GPU.

**2:24.4  Try it**  
Everything is in the browser: the video, the replay, the robot and the measured model, with URDF downloads. One video in. A model that moves like the real thing out.
