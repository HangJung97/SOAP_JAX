# NNX / Optax compatibility

This fork preserves parameter PyTrees through direct Optax updates and
`nnx.Optimizer`, including NNX variable metadata and Equinox static fields.
Initialize SOAP with the original parameters, supply gradients with the same
structure, and apply the returned updates with `optax.apply_updates`:

```python
optimizer = soap(learning_rate=1e-3)
optimizer_state = optimizer.init(params)
updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
params = optax.apply_updates(params, updates)
```

SOAP uses a consistent tuple of array leaves internally and reconstructs updates
with the incoming gradient tree definition. Moments and preconditioners remain
stable across compiled calls without storing custom preconditioners inside NNX
variable values. No framework-specific optimizer adapter is required.

The implementation starts from SOAP_JAX `v0.2.2`
(`a7c92f174c49b13136294a94ae0d46266b4f671f`). Arithmetic, schedule indexing,
weight decay, bias correction, the initial zero update, and preconditioner
refresh timing are unchanged.

## Checkpoint compatibility

**The optimizer-state layout changes.** Moments, `GG`, and `Q` now use tuples
in JAX leaf order for all parameter representations. Existing `v0.2.2`
full-state checkpoints are not generally compatible.

Load model weights and initialize a new optimizer, or explicitly migrate each
old optimizer field to the corresponding tuple. There is no automatic migration.
Reinitializing the optimizer does not exactly resume an old run.

For new checkpoints, restore into a fresh matching target state and retain the
same model structure, optimizer configuration, and dependency versions. The
validation below checks exact continuation on each backend. It does not establish
CPU/GPU cross-backend equality or multi-GPU support.

## Implementation note

[PR #4](https://github.com/haydn-jones/SOAP_JAX/pull/4), already included in
`v0.2.2`, unwraps NNX variables only during initialization to prevent custom
preconditioner corruption inside `nnx.Optimizer`. On Flax 0.12.9 that wrapper
also converts gradients to arrays before calling Optax. Direct Optax calls on
`nnx.split` state retain `Param` nodes in gradients, so initialization and updates
still disagree about the tree structure. PR #4 fixes the wrapper path but does
not cover this direct path.

Removing initialization-only unwrapping fixes direct Optax but exposes the
preconditioner corruption again in `nnx.Optimizer`. A named-tuple replacement
alone also fails. Consistent array conversion at both optimizer boundaries
supports both paths while retaining the existing preconditioner class.
[PR #3](https://github.com/haydn-jones/SOAP_JAX/pull/3)'s tuple/mapping conversion
is unnecessary with this approach; none of its dtype or counter changes are
included.

## Validation

The committed `uv.lock` pins the test environment. Python 3.12+ is required for
the development group; library Python support is unchanged. Core test versions
are JAX/JAXlib 0.11.1, Flax 0.12.9, Optax 0.2.8, Orbax 0.12.4, and Equinox 0.13.8.

The regression suite compares identical initial parameters and supplied gradients
across dictionaries, NNX, Equinox, and an independent `v0.2.2` implementation.
It checks matrix/vector/scalar leaves, both `precondition_1d` settings, excluded
dimensions, float32/float64, finite states, exact update trees, the initial zero
update, nine ordinary steps, four refreshes, schedules, and weight decay.
`nnx.Optimizer` has a separate parity test.

### CPU regression tests

```bash
uv sync --locked --python 3.12
git show v0.2.2:src/soap_jax/soap.py > /tmp/soap-v022.py
JAX_PLATFORMS=cpu SOAP_BASELINE=/tmp/soap-v022.py uv run --no-sync pytest tests/test_nnx.py tests/test_parity.py
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
```

To reproduce the failure against the untouched baseline, run:

```bash
mkdir -p /tmp/soap-baseline/soap_jax
cp /tmp/soap-v022.py /tmp/soap-baseline/soap_jax/soap.py
cp src/soap_jax/__init__.py /tmp/soap-baseline/soap_jax/__init__.py
JAX_PLATFORMS=cpu PYTHONPATH=/tmp/soap-baseline uv run --no-sync pytest tests/test_nnx.py -k direct_optax
```

The baseline's direct NNX Optax test fails with a `Param`/`Preconditioner` tree
mismatch. Its array control passes on the pinned versions.

### PhiJAX integration and checkpoint tests

Integration validation uses PhiJAX 0.2.0b4 at commit
`6b4ba66f1587498ec778ef766f9cb9291141b1a4`, through its existing Optax interface
without a Trainer change:

```bash
git clone https://github.com/HangJung97/PhiJAX /tmp/phijax-soap-validation
git -C /tmp/phijax-soap-validation checkout 6b4ba66f1587498ec778ef766f9cb9291141b1a4
uv pip install --python .venv/bin/python /tmp/phijax-soap-validation
JAX_PLATFORMS=cpu PHIJAX_SOURCE=/tmp/phijax-soap-validation uv run --no-sync pytest tests/test_phijax.py
```

The MLP uses the actual compiled `make_train_step`. It saves with
`OrbaxCheckpointIO` after seven steps and compares every state and metric for
eight further steps after restore. The heat test uses the quickstart's equation,
DataModule, MLP with two 32-unit hidden layers, and Trainer. It compares 16
uninterrupted steps with seven saved plus nine resumed steps.

The refresh frequency is two. All `TrainState` fields, including optimizer slots
and random keys, must match exactly. These are compatibility smoke tests, not
claims of converged PDE accuracy.

### CUDA 13 and recorded results

Sync the optional CUDA 13 group before installing the integration checkout, then
repeat the tests on the GPU:

```bash
uv sync --locked --group cuda13
uv pip install --python .venv/bin/python /tmp/phijax-soap-validation
JAX_PLATFORMS=cuda PHIJAX_SOURCE=/tmp/phijax-soap-validation SOAP_BASELINE=/tmp/soap-v022.py uv run --no-sync pytest tests
```

All 22 compatibility tests passed on CPU and on an NVIDIA RTX PRO 2000 Blackwell
Generation Laptop GPU (driver 596.53), with the locked JAX CUDA 13 environment.
Both backends passed exact baseline/representation comparisons and full-state
checkpoint continuation.
