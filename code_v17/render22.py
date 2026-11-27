"""v22 render module for sim/export_mjcf.py (KITORU_VER=22): the render21 chains and nod_pose on the current
build21_arm geometry. build21_arm.build_model() now starts from build21_body.build_model() (torso / head print fits
and the second pass: shell walls 2.0, dovetails, screws) which starts from build21_head / build20.build_model()
(legs incl. xl_horn_fit, fdm_joints, assembly_durability, print_robust), then adds the elbow arm with its durability
pass. Mass: sim/model/mass_kitoru_v22_light.json (sim/tools/mass_v22.py, every link voxelised from this geometry).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render21 as _R21

r = _R21.r
nod_pose = _R21.nod_pose
