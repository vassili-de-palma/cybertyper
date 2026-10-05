"""One-off calibration tools.

* ``click_homography``     the calibration used by the visual-servo evals:
                           lock a scan pose, click points in the camera
                           image, touch them with the gripper, fit a
                           pixel -> robot-XY homography and record the
                           table plane height. Writes
                           ``configs/click_homography.json``.
* ``session_calibration``  ChArUco intrinsics + hand-eye + plane probe for
                           the legacy open-loop pipeline. Writes
                           ``configs/rig_calibration.json``.
* ``press_depth``          per-keyboard key-travel sample for the legacy
                           open-loop pipeline. Writes
                           ``configs/press_depth.json``.
"""
