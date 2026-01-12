VCX Lab 4 - Motion Matching Patch

How to apply:
1) Copy the folders/files in this patch into the root of your vci-2025 repo (overwrite).
2) Rebuild with xmake:
   xmake f -m debug
   xmake -v lab4
   xmake run lab4

What this patch contains:
- Lab4 now contains only the Motion Matching case (App/UI updated).
- Added a minimal BVH loader (BVHLoader.*) supporting per-joint Euler channels.
- Implemented classic motion-matching database + search (MotionMatching.*).
- Implemented Motion Matching demo case with a procedural skinned mesh + skeleton/trajectory debug (CaseMotionMatching.*).
- Added shader: assets/shaders/skinned.vert|frag.
- Added demo mocap clip: assets/mocap/mm_synthetic_walk_turn.bvh.

Controls (default):
- W/S: forward/back
- A/D: turn left/right
- Shift: run
- Mouse: orbit camera (LMB drag), zoom (wheel)

