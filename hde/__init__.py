"""HD-EPIC-NF — Experiment B of the NF forget/mistake project.

A conditional Normalizing Flow over *where objects are handled in a real kitchen*:

    log p( 3-D location | kitchen )

Unlike Experiment A (RoomR), a real kitchen's object-location density is strongly
MULTIMODAL (hob, sink, counters, fridge, storage), so this is where a Flow should
beat a Gaussian mixture -- the direct answer to A's "a Gaussian tied us" lesson.
We also use HD-EPIC's eye-gaze to ask whether surprising placements are the ones
made away from where the person was looking. Data = HD-EPIC open annotations
(eye-gaze priming: 3-D object location + gaze point); no video needed.
"""
