# Reference Validation

This directory is reserved for tests that compare encoder outputs with either
independent implementations or direct numerical definitions. Tests here should
make scientific claims explicit: what is compared, why the comparison is valid,
and what numerical differences are expected from normalization or boundary
handling.

Validation levels are recorded in `tscv_vision.representations.metadata` and
rendered into `docs/encoder_validation.md`. Adding a reference test is not
enough by itself; update the metadata only when the test genuinely supports the
stronger claim.
