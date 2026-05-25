import flax.linen as nn
import jax.numpy as jnp

class Temperature(nn.Module):
    initial_temperature: float = 0.01

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)