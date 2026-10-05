"""Legacy open-loop entry points (superseded by the visual-servo evals).

These wrap :mod:`cybertyping.tasks.session`: one overhead scan, OCR anchors,
homography, then a *calibrated* pixel-to-robot projection and an IK-planned
press with no camera in the loop. The calibration never reached the
required accuracy (see the README); the scripts are
kept as a reference implementation and still run in dry-run mode::

    CYBERTYPING_DRY_RUN=1 python -m evals.open_loop.eval3_sentence --type "hi" --image frame.jpg
"""
