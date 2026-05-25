from math import sqrt

from flax import linen as nn
import jax.numpy as jnp
import jax

from typing import Type

from jaxrl5.networks.mlp import MLP

class FourierFeatures(nn.Module):
    output_size: int
    learnable: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_size // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_size // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    frequency_embedding_size: int = 128
    fourier_feature_learnable: bool = True
    hidden_size: int = 64
    activations: Type[nn.Module] = nn.gelu

    @nn.compact
    def __call__(self, t):
        # t_freq = self.fourier_features()(t)
        # t_emb = self.mlp(t_freq)
        t_freq = FourierFeatures(
            output_size=self.frequency_embedding_size,
            learnable=self.fourier_feature_learnable)(t)
        t_emb = MLP(
            hidden_dims=(self.hidden_size, self.hidden_size),
            activations=self.activations,
            activate_final=False)(t_freq)
        return t_emb
    
class MeanFlow(nn.Module):
    t_embedder_cls: Type[nn.Module]
    h_embedder_cls: Type[nn.Module]
    mean_velocity_cls: Type[nn.Module]

    @nn.compact
    def __call__(self,
                s: jnp.ndarray,
                a: jnp.ndarray,
                t: jnp.ndarray,
                h: jnp.ndarray,
                training: bool = False):

        t_embed = self.t_embedder_cls()(t)
        h_embed = self.h_embedder_cls()(h)
        cond = t_embed + h_embed
        input = jnp.concatenate([a, s, cond], axis=-1)

        return self.mean_velocity_cls()(input, training=training)
    
def calc_normal_logprob(noise):
    """Calculate log probability of noise under standard normal distribution."""
    dim = noise.shape[-1]
    log_z = 0.5 * dim * jnp.log(2 * jnp.pi)
    logp = -0.5 * jnp.sum(noise ** 2, axis=-1) - log_z
    return logp
    
def action_sampler(actor_fn, params, T, noise, observations, clip_sampler, t_min = 0.0, t_max = 1.0, training = False):
    B = noise.shape[0]
    time_schedule = jnp.linspace(t_min, t_max, T+1)

    def step_fn(inputs, i):
        current_x = inputs
        t = jnp.expand_dims(jnp.array([time_schedule[i]], dtype=jnp.float32).repeat(B), axis=1)
        h = jnp.expand_dims(jnp.array([time_schedule[i] - time_schedule[i - 1]], dtype=jnp.float32).repeat(B), axis=1)

        u_pred = actor_fn(
            {"params": params}, 
            observations, current_x, t, h, training=training
        )
        
        current_x = current_x - u_pred * h

        return current_x, None

    action_0, _ = jax.lax.scan(
        step_fn, noise,
        jnp.arange(T, 0, -1), unroll=2)
    action_0 = jnp.clip(action_0, -1, 1) if clip_sampler else action_0

    return action_0

def action_sampler_with_logp(actor_fn, actor_params, logp_fn, logp_params, T, noise, observations, clip_sampler, t_min = 0.0, t_max = 1.0, training = False):
    B = noise.shape[0]
    time_schedule = jnp.linspace(t_min, t_max, T+1)

    def step_fn(inputs, i):
        (current_x, current_logp) = inputs
        t = jnp.expand_dims(jnp.array([time_schedule[i]], dtype=jnp.float32).repeat(B), axis=1)
        h = jnp.expand_dims(jnp.array([time_schedule[i] - time_schedule[i - 1]], dtype=jnp.float32).repeat(B), axis=1)

        u_pred = actor_fn(
            {"params": actor_params}, 
            observations, current_x, t, h, training=training
        )
        logp_mvel_pred = logp_fn(
            {"params": logp_params},
            observations, current_x, t, h, training=training
        )
        
        current_x = current_x - u_pred * h
        current_logp = current_logp + logp_mvel_pred * h
        outputs = (current_x, current_logp)

        return outputs, None

    noise_logp = jnp.expand_dims(calc_normal_logprob(noise), axis=-1)
    (action_0, current_logp), _ = jax.lax.scan(
        step_fn, (noise, noise_logp),
        jnp.arange(T, 0, -1), unroll=2)
    action_0 = jnp.clip(action_0, -1, 1) if clip_sampler else action_0
    current_logp = jnp.squeeze(current_logp, axis=-1)

    return action_0, current_logp

def _logit_normal_dist(rng, batch_size):
    P_std = 1.0
    P_mean = -0.4
    rnd_normal = jax.random.normal(rng, [batch_size, 1], dtype=jnp.float32)
    return nn.sigmoid(rnd_normal * P_std + P_mean)
  
def _uniform_dist(rng, batch_size):
    return jax.random.uniform(rng, [batch_size, 1], dtype=jnp.float32)

def tr_sampler(rng, batch_size: int, time_dist_name, data_proportion: float):
    if time_dist_name == 'logit_normal':
        time_dist = _logit_normal_dist
    elif time_dist_name == 'uniform':
        time_dist = _uniform_dist
    else:
        raise ValueError(f"Invalid time_dist_name: {time_dist_name}")
    key, rng = jax.random.split(rng, 2)
    t = time_dist(key, batch_size)
    key, rng = jax.random.split(rng, 2)
    r = time_dist(key, batch_size)
    t, r = jnp.maximum(t, r), jnp.minimum(t, r)

    data_size = int(batch_size * data_proportion)
    zero_mask = jnp.arange(batch_size) < data_size
    zero_mask = zero_mask.reshape(batch_size, 1)
    r = jnp.where(zero_mask, t, r)

    return t, r