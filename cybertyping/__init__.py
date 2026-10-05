"""cybertyping: an SO-101 arm that types on a keyboard it has never seen.

Package layout::

    core/          hardware-validated kinematics, IK, streaming, contact descent
    servo/         closed-loop visual servoing (the demo-day pipeline)
    perception/    camera capture, dot/OCR/VLM detectors, layout + homography
    calibration/   click-homography scan-pose capture (+ legacy rig calibration)
    analytic_ik/   closed-form IK cross-check
    tasks/, press/, policy/
                   legacy open-loop pipeline (see README, "Earlier approaches")

Entry points live in ``evals/`` (one per rubric item) and ``scripts/``
(calibration, diagnostics, demos). See README.md.
"""

__version__ = "1.0.0"
