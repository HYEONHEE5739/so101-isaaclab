# Isaac Lab Mimic portable reference

The five datagen modules are from the installed Isaac Lab Mimic **1.0.15**.
`UPSTREAM.json` records exact upstream paths/hashes. Their only changes are import
destinations. Source selection, subtask segmentation/offsets, object-relative
transforms, waypoint stitching/interpolation, noise and generation bodies remain
unchanged. Both backend adapters execute this same reference code.

The configuration definitions are copied from the installed `mimic_env_cfg.py`;
their construction uses a portable deep-copy decorator. The math module contains
the transitive function subset needed by the SO101 single-arm configuration, preserving function
bodies and torch decorators. Dataset I/O uses the shared HDF5/legacy reader.

Original copyright and SPDX headers are retained. See `LICENSE` (BSD-3-Clause)
and `LICENSE-mimic` (the upstream Mimic licensing terms). This directory is an
explicitly tracked port, not an import of Isaac/Omniverse in the MuJoCo process.
Updates require the upstream-body equivalence tests and recorded version review.
This adapter exposes the SO101 two-subtask configuration with SkillGen disabled;
multi-arm coordination and SkillGen are not supported by this portable adapter.
