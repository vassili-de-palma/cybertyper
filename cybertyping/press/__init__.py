"""Press-motion strategies for the legacy open-loop pipeline.

Both modules expose the same callable::

    descend.press(chain, robot, pre_tip_xyz, q_pre, tip_off, gripper_pos)
        -> (contact_q | target_q, descent_qs, info_dict)

* ``descend.load_contact``  contact-sensed descent (default)
* ``descend.open_loop``     fixed depth from ``configs/press_depth.json``

The visual-servo pipeline in :mod:`cybertyping.servo` has its own descent
(:func:`cybertyping.servo.control.descend_until_contact`) and does not use
this package.
"""
