"""Closed-loop visual servoing for the SO-101 wrist camera.

This is the approach that ran on demo day. Instead of converting pixels to
robot coordinates through a calibrated camera model (see
the README for why that was abandoned), the loop is
closed in image space:

1. A coloured marker on the gripper tip is detected in every frame.
2. A *target* provides the pixel the tip should reach. Targets are
   pluggable (:mod:`cybertyping.servo.targets`):

   * ``ColorDotTarget``  a coloured dot on paper (Eval 1)
   * ``KeyTarget``       a keyboard key, located by OCR or a VLM plus a
                         homography to the canonical US layout (Eval 2/3)

3. The pixel error is mapped to a small robot-XY step through a fixed 2x2
   gain and executed with the position Jacobian
   (:mod:`cybertyping.servo.control`).
4. A press is three phases (:mod:`cybertyping.servo.press`): diagonal
   approach while descending to a hover height, planar refinement until the
   pixel error is below tolerance, then a straight vertical descent until
   the shoulder motor load reports contact.

Module map::

    camera.py    cross-platform OpenCV capture helpers
    rig.py       Rig: robot + kinematic chain + camera + saved scan pose
    targets.py   tip detection and the pluggable target providers
    control.py   servo loop, orientation hold, descents
    press.py     press_target / hover_target recipes used by evals/
    cli.py       shared argparse flags for the eval entry points
"""

from .rig import Rig, ScanPose, load_scan_pose
from .targets import ColorDotTarget, KeyTarget, find_tip
from .press import hover_target, press_target, wait_for_start

__all__ = [
    "Rig", "ScanPose", "load_scan_pose",
    "ColorDotTarget", "KeyTarget", "find_tip",
    "hover_target", "press_target", "wait_for_start",
]
