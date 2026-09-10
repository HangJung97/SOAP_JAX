import jax
import pytest


@pytest.fixture(autouse=True)
def enable_float64():
    # Integration tests select their own precision; restore x64 for each dtype regression.
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)
