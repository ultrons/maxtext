"""Split-phase (start/done) DMA wrappers.

`start` arms a DMA and returns without waiting, so XLA can schedule real compute after it.
`done` RECONSTRUCTS the identical descriptor and waits. Both are ordinary TC-shaped custom
calls, so XLA schedules around them rather than fencing at them.

No semaphore crosses the kernel boundary. Each half allocates its own DMA semaphore as
kernel scratch, which is what places it in sync-flag memory; because Mosaic's allocation is
deterministic, `done`'s rebuilt descriptor names the sync flag `start` armed.

HARD RULE, measured on v7x (probes/startdone_runtime.py, gates R5/R6): the two halves must
declare BYTE-IDENTICAL `scratch_shapes`. Under a differing scratch footprint Mosaic assigns
the DMA semaphore a different slot, the wait targets the wrong flag, and it HANGS -- it does
not raise. `_SCRATCH` below is shared by both halves for exactly this reason, and
`assert_scratch_matches()` is the guard so a future edit cannot break it silently.
"""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_HBM = pl.BlockSpec(memory_space=pltpu.HBM)

# The ONE scratch signature. Both halves must use this exact list (see R5/R6).
_SCRATCH = [pltpu.SemaphoreType.DMA]


def assert_scratch_matches(a, b):
  """Guard the R5 hang: identical scratch signatures on both halves."""
  if list(a) != list(b):
    raise ValueError(
        "start/done scratch_shapes must be byte-identical; mismatched footprints make "
        f"Mosaic slot the DMA semaphore differently and the wait HANGS. got {a} vs {b}"
    )


def _start_body(x_ref, o_ref, sem):
  pltpu.make_async_copy(x_ref, o_ref, sem).start()


def _done_body(x_ref, d_ref, o_ref, sem):
  # d_ref is aliased to the output, so nothing is loaded here (Mosaic forbids HBM loads).
  pltpu.make_async_copy(x_ref, d_ref, sem).wait()


def _split_copy_impl(x):
  """Identity, expressed as an armed DMA and a separate wait.

  Structurally this is what a split-phase gather looks like from XLA's point of view: two
  TC custom calls with a real buffer dependency between them, and room for compute in the
  gap. It moves the same bytes a plain copy would, so it is a placement probe rather than a
  performance change.
  """
  assert_scratch_matches(_SCRATCH, _SCRATCH)
  shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
  start = pl.pallas_call(
      _start_body, in_specs=[_HBM], out_specs=_HBM, out_shape=shape,
      scratch_shapes=_SCRATCH,
  )
  done = pl.pallas_call(
      _done_body, in_specs=[_HBM, _HBM], out_specs=_HBM, out_shape=shape,
      scratch_shapes=_SCRATCH, input_output_aliases={1: 0},
  )
  return done(x, start(x))


# A pallas_call on a grad-live path cannot be differentiated: autodiff descends into the
# kernel and Pallas's JVP rule asserts (`_pallas_call_jvp_rule` -> `ad.jvp_jaxpr` ->
# AssertionError). Wrap it so autodiff never enters, and supply the transpose ourselves --
# the same reason `moe.py`'s `_make_cv_gather` is a custom_vjp. The PRIMAL keeps the real
# start/done pair, so `remat_policy=custom` re-runs the pair in the backward, which is the
# property this probe exists to measure.
@jax.custom_vjp
def split_copy(x):
  return _split_copy_impl(x)


def _split_copy_fwd(x):
  return _split_copy_impl(x), None


def _split_copy_bwd(_res, ct):
  # Transpose of an identity copy is the identity. A real gather's transpose is a tiled
  # psum_scatter (see `_make_cv_gather`).
  return (ct,)


split_copy.defvjp(_split_copy_fwd, _split_copy_bwd)
