# Video2URDF: details

The long material, moved out of the [README](../README.md) to keep it short:

- [RESULTS.md](RESULTS.md): everything measured and how it was scored. Real objects, the RBO
  motion-capture benchmark, synthetic ground truth, repeat runs and objects it had never seen,
  open weights only, the no-tools baseline, URDF Doctor, physics, shape design, completing an
  unseen body, the robot tasks, plain video without LiDAR; and, outside the pipeline, a drawer
  policy experiment that is not part of the submission.
- [HOW_IT_WORKS.md](HOW_IT_WORKS.md): the pipeline, the two capture paths, which Meta model
  decides what, how SAM 3.1 tracks the parts, a map of the code, and how it differs from
  research on articulated objects.
- [COMMANDS.md](COMMANDS.md): every command, by task.
- [spark_alone.md](spark_alone.md): the same Muse Spark asked plainly for a URDF, no tools:
  table, caveats and all 30 answers.
- [demo_script.md](demo_script.md): the demo video's narration, with timings.
- [appearance_validation.md](appearance_validation.md): the development log of the experimental
  appearance-completion path.
- [scripts/README.md](../scripts/README.md): the scripts, by purpose.
