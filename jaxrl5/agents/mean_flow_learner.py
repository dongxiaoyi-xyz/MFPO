"""Implementations of algorithms for continuous control."""
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
import optax
from flax.training.train_state import TrainState
from flax import struct

from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import MLP, Temperature
from jaxrl5.networks.mean_flow import MeanFlow, TimestepEmbedder
from jaxrl5.networks.mean_flow import action_sampler, action_sampler_with_logp, tr_sampler
from jaxrl5.networks.state_action_value import DistributionalStateActionValue

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)


def tensorstats(tensor, prefix=None):
  assert tensor.size > 0, tensor.shape
  metrics = {
      'mean': tensor.mean(),
      'std': tensor.std(),
      'mag': jnp.abs(tensor).max(),
      'min': tensor.min(),
      'max': tensor.max(),
  }
  if prefix:
    metrics = {f'{prefix}_{k}': v for k, v in metrics.items()}
  return metrics

def categorical_projection(
    next_prob,      # [B, N]        # [B,]
    tz,          # [B,N]
    v_min,
    delta_z,
):
    B, N = next_prob.shape

    b = (tz - v_min) / delta_z
    l = jnp.floor(b).astype(jnp.int32)
    u = jnp.ceil(b).astype(jnp.int32)

    l = jnp.clip(l, 0, N - 1)
    u = jnp.clip(u, 0, N - 1)

    proj = jnp.zeros((B, N))

    batch_idx = jnp.arange(B)[:, None]

    eq = (l == u)

    proj = proj.at[batch_idx, l].add(
        next_prob * jnp.where(eq, 1.0, u - b)
    )
    proj = proj.at[batch_idx, u].add(
        next_prob * jnp.where(eq, 0.0, b - l)
    )

    return proj

class MeanFlowLearner(Agent):
    actor: TrainState
    logp_mvel: TrainState
    critic_1: TrainState
    critic_2: TrainState
    target_critic_1: TrainState
    target_critic_2: TrainState
    temp: TrainState

    discount: float
    tau: float
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    clip_sampler: bool = struct.field(pytree_node=False)
    backup_entropy: bool = struct.field(pytree_node=False)
    vel1_samples_num: int = struct.field(pytree_node=False)
    vel2_samples_num: int = struct.field(pytree_node=False)
    eval_action_selection : bool
    eval_candidate_num: int = struct.field(pytree_node=False)
    div_samples_num: int = struct.field(pytree_node=False)
    time_dist_name: str = struct.field(pytree_node=False)
    data_proportion: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    actor_delay: int = struct.field(pytree_node=False)
    use_cdq: bool = struct.field(pytree_node=False)
    delta_z: Optional[float] = struct.field(pytree_node=False)
    v_min: Optional[float] = struct.field(pytree_node=False)
    v_max: Optional[float] = struct.field(pytree_node=False)
    z_atoms: Optional[jnp.ndarray] = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_architecture: str = 'mlp',
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        logp_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        actor_layer_norm: bool = True,
        critic_layer_norm: bool = True,
        T: int = 2,
        time_dim: int = 128,
        clip_sampler: bool = True,
        temp: float = 1,
        backup_entropy: bool = True,
        vel1_samples_num: int = 16,
        vel2_samples_num: int = 32,
        eval_action_selection : bool = True,
        eval_candidate_num: int = 10,
        div_samples_num: int = 2,
        time_dist_name: str = 'logit_normal',
        data_proportion: float = 0.75,
        target_entropy_coeff: float = -0.5,
        actor_delay: int = 1,
        use_cdq: bool = True,
        atoms_num: int = 101,
        v_min: float = -1600.0,
        v_max: float = 1600.0,
    ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, logp_key, temp_key = jax.random.split(rng, 5)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[-1]

        # Time embedding network.
        t_embedder_cls = partial(TimestepEmbedder, 
                                 frequency_embedding_size=time_dim, fourier_feature_learnable=True, 
                                 hidden_size=64, activations=nn.gelu)
        r_embedder_cls = partial(TimestepEmbedder, 
                                 frequency_embedding_size=time_dim, fourier_feature_learnable=True, 
                                 hidden_size=64, activations=nn.gelu)
        

        if actor_architecture == 'mlp':
            mean_velocity_cls = partial(MLP,
                hidden_dims=tuple(list(actor_hidden_dims) + [action_dim]),
                activations=nn.gelu, use_layer_norm=actor_layer_norm,
                activate_final=False)
            
            actor_def = MeanFlow(
                t_embedder_cls=t_embedder_cls,
                h_embedder_cls=r_embedder_cls,
                mean_velocity_cls=mean_velocity_cls,
            )

            mean_div_cls = partial(MLP,
                hidden_dims=tuple(list(actor_hidden_dims) + [1]),
                activations=nn.gelu, use_layer_norm=actor_layer_norm,
                activate_final=False)
            
            logp_mvel_def = MeanFlow(
                t_embedder_cls=t_embedder_cls,
                h_embedder_cls=r_embedder_cls,
                mean_velocity_cls=mean_div_cls,
            )
        else:
            raise ValueError(f'Invalid actor architecture: {actor_architecture}')
        
        time = jnp.ones((1, 1))
        observations = jnp.expand_dims(observations, axis = 0)
        actions = jnp.expand_dims(actions, axis = 0)
        actor_params = actor_def.init(
            actor_key, observations, actions, time, time)['params']
        actor = TrainState.create(
            apply_fn=actor_def.apply, params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),)
        logp_params = logp_mvel_def.init(
            logp_key, observations, actions, time, time)['params']
        logp_mvel = TrainState.create(
            apply_fn=logp_mvel_def.apply, params=logp_params,
            tx=optax.adam(learning_rate=logp_lr))

        # Initialize critics.
        critic_base_cls = partial(
            MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=critic_layer_norm, activations=nn.gelu)
        critic_def = DistributionalStateActionValue(critic_base_cls, atoms_num=atoms_num)
        critic_key_1, critic_key_2 = jax.random.split(critic_key, 2)
        critic_params_1 = critic_def.init(critic_key_1, observations, actions)["params"]
        critic_params_2 = critic_def.init(critic_key_2, observations, actions)["params"]
        critic_1 = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params_1,
            tx=optax.adam(learning_rate=critic_lr),)
        critic_2 = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params_2,
            tx=optax.adam(learning_rate=critic_lr))

        target_critic_def = DistributionalStateActionValue(critic_base_cls, atoms_num=atoms_num)
        target_critic_1 = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params_1,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),)
        target_critic_2 = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params_2,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),)

        target_entropy = target_entropy_coeff * action_dim
        temp_def = Temperature(temp)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        z_atoms = jnp.linspace(v_min, v_max, atoms_num)
        delta_z = (v_max - v_min) / (atoms_num - 1)

        return cls(
            actor=actor,
            logp_mvel=logp_mvel,
            critic_1=critic_1,
            critic_2=critic_2,
            target_critic_1=target_critic_1,
            target_critic_2=target_critic_2,
            tau=tau,
            discount=discount,
            rng=rng,
            act_dim=action_dim,
            T=T,
            clip_sampler=clip_sampler,
            temp=temp,
            backup_entropy=backup_entropy,
            vel1_samples_num=vel1_samples_num,
            vel2_samples_num=vel2_samples_num,
            eval_action_selection=eval_action_selection,
            eval_candidate_num=eval_candidate_num,
            div_samples_num=div_samples_num,
            time_dist_name=time_dist_name,
            data_proportion=data_proportion,
            target_entropy=target_entropy,
            actor_delay=actor_delay,
            use_cdq=use_cdq,
            delta_z=delta_z,
            v_min=v_min,
            v_max=v_max,
            z_atoms=z_atoms,
        )

    def update_q(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        (B, _) = batch['observations'].shape
        (_, A) = batch['actions'].shape

        # Sample actions for next state.
        key, rng = jax.random.split(agent.rng)
        noise = jax.random.normal(
            key, (B, agent.act_dim))
        next_actions, next_logps = action_sampler_with_logp(
            agent.actor.apply_fn,
            agent.actor.params,
            agent.logp_mvel.apply_fn,
            agent.logp_mvel.params,
            agent.T, noise,
            batch['next_observations'],
            agent.clip_sampler)
        assert next_actions.shape == (B, A)

        # Compute target q.
        next_q_1 = agent.target_critic_1.apply_fn(
            {"params": agent.target_critic_1.params}, batch["next_observations"], next_actions)
        next_q_2 = agent.target_critic_2.apply_fn(
            {"params": agent.target_critic_2.params}, batch["next_observations"], next_actions)
        next_q_atoms_1 = jnp.repeat(agent.z_atoms[None, :], axis=0, repeats=B)  # [B, N]
        next_q_atoms_2 = jnp.repeat(agent.z_atoms[None, :], axis=0, repeats=B)  # [B, N]

        temp = agent.temp.apply_fn({"params": agent.temp.params})

        # compute log probality
        if agent.backup_entropy:
            next_q_atoms_1 = next_q_atoms_1 - (temp * next_logps)[:, None]
            next_q_atoms_2 = next_q_atoms_2 - (temp * next_logps)[:, None]

        target_q_atoms_1 = batch["rewards"][:, None] + agent.discount * batch["masks"][:, None] * next_q_atoms_1
        target_q_atoms_1 = jnp.clip(target_q_atoms_1, agent.v_min, agent.v_max)
        target_q_atoms_2 = batch["rewards"][:, None] + agent.discount * batch["masks"][:, None] * next_q_atoms_2
        target_q_atoms_2 = jnp.clip(target_q_atoms_2, agent.v_min, agent.v_max)
        target_q_1 = categorical_projection(
            jax.nn.softmax(next_q_1, axis=-1),
            target_q_atoms_1,
            agent.v_min,
            agent.delta_z,
        )  # [B, N]
        target_q_2 = categorical_projection(
            jax.nn.softmax(next_q_2, axis=-1),
            target_q_atoms_2,
            agent.v_min,
            agent.delta_z,
        )  # [B, N]
        target_q_value_1 = jnp.sum(target_q_1 * agent.z_atoms[None, :], axis=-1)
        target_q_value_2 = jnp.sum(target_q_2 * agent.z_atoms[None, :], axis=-1)
        target_q_min = jnp.where((target_q_value_1 < target_q_value_2)[:, None], target_q_1, target_q_2)
        target_q_mean = (target_q_1 + target_q_2) / 2
        target_q = target_q_min if agent.use_cdq else target_q_mean
        metrics = {}
        metrics.update({'next_logps': next_logps.mean()})
        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            q = agent.critic_1.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"])
            log_prob = jax.nn.log_softmax(q, axis=-1)
            loss = -jnp.sum(sg(target_q) * log_prob, axis=-1)
            loss = loss.mean()
            metrics = {**tensorstats(jnp.sum(jax.nn.softmax(q) * agent.z_atoms[None], axis=-1), 'q')}
            metrics.update({'c_loss': loss})
            return loss, metrics

        grads_c_1, metrics_c_1 = jax.grad(critic_loss_fn, has_aux=True)(agent.critic_1.params)
        metrics.update({f'{k}_1': v for k, v in metrics_c_1.items()})
        critic_1 = agent.critic_1.apply_gradients(grads=grads_c_1)

        grads_c_2, metrics_c_2 = jax.grad(critic_loss_fn, has_aux=True)(agent.critic_2.params)
        metrics.update({f'{k}_2': v for k, v in metrics_c_2.items()})
        critic_2 = agent.critic_2.apply_gradients(grads=grads_c_2)

        target_critic_1_params = optax.incremental_update(
            critic_1.params, agent.target_critic_1.params, agent.tau)
        target_critic_2_params = optax.incremental_update(
            critic_2.params, agent.target_critic_2.params, agent.tau)
        target_critic_1 = agent.target_critic_1.replace(params=target_critic_1_params)
        target_critic_2 = agent.target_critic_2.replace(params=target_critic_2_params)
        new_agent = agent.replace(
            critic_1=critic_1, critic_2=critic_2,
            target_critic_1=target_critic_1,
            target_critic_2=target_critic_2,
            rng=rng)
        return new_agent, metrics

    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        B, A = batch['actions'].shape
        key, rng = jax.random.split(agent.rng, 2)
        t, r = tr_sampler(key, B, agent.time_dist_name, agent.data_proportion)

        K1 = agent.vel1_samples_num
        K2 = agent.vel2_samples_num

        temp = agent.temp.apply_fn({"params": agent.temp.params})

        def get_policy_actions(rng, K1):
            key, rng = jax.random.split(rng, 2)
            init_noise = jax.random.normal(key, (B * K1, agent.act_dim))
            policy_actions, logps = action_sampler_with_logp(
                agent.actor.apply_fn,
                agent.actor.params,
                agent.logp_mvel.apply_fn,
                agent.logp_mvel.params,
                agent.T, init_noise,
                jnp.repeat(batch['observations'], axis=0, repeats=K1),
                agent.clip_sampler)
            policy_actions = policy_actions.reshape(B, K1, agent.act_dim) # [B, K1, A]
            logps = logps.reshape(B, K1)
            return policy_actions, logps
    
        def get_clean_actions(rng, noisy_actions, K2):
            noisy_actions_repeat = jnp.repeat(jnp.expand_dims(noisy_actions, axis=1), axis=1, repeats=K2) # [B, K2, A]
            lower_bound = (noisy_actions_repeat - (1 - t)[:, :, None]) / t[:, :, None]
            upper_bound = (noisy_actions_repeat + (1 - t)[:, :, None]) / t[:, :, None]
            key, rng = jax.random.split(rng, 2)
            tnormal_noise = jax.random.truncated_normal(
                key, lower=lower_bound, upper=upper_bound, shape=(B, K2, agent.act_dim))
            key, rng = jax.random.split(rng, 2)
            normal_noise = jax.random.normal(key, shape=((B, K2, agent.act_dim)))
            normal_noise_clip = jnp.clip(normal_noise, min=lower_bound, max=upper_bound)
            # jax.random.truncated_normal() generates NaN occasionally, so use clipped normal noise to replace NaN
            noise_samples = jnp.where(jnp.isnan(tnormal_noise), normal_noise_clip, tnormal_noise)
            clean_actions = (noisy_actions_repeat - t[:, :, None] * noise_samples) / (1 - t)[:, :, None]

            return clean_actions
        
        key, rng = jax.random.split(rng, 2)
        policy_actions, logps_policy_actions = get_policy_actions(key, K1)
        key, rng = jax.random.split(rng, 2)
        noise = jax.random.normal(
            key, (B, agent.act_dim))
        noisy_actions = (1 - t) * policy_actions[:, 0] + t * noise
        key, rng = jax.random.split(rng, 2)
        clean_actions = get_clean_actions(key, noisy_actions, K2)

        devices = jax.devices()
        assert B % len(devices) == 0

        # Compute Q
        # @partial(shard_map, mesh=Mesh(devices, ('i',)), in_specs=(P('i'), P('i')), out_specs=(P('i')))
        def compute_Q(actions, observations):
            critic_dist_1 = agent.critic_1.apply_fn(
                {"params": agent.critic_1.params}, observations, actions)
            critic_dist_2 = agent.critic_2.apply_fn(
                {"params": agent.critic_2.params}, observations, actions)
            critic_1 = jnp.sum(jax.nn.softmax(critic_dist_1, axis=-1) * agent.z_atoms[None, None], axis=-1)
            critic_2 = jnp.sum(jax.nn.softmax(critic_dist_2, axis=-1) * agent.z_atoms[None, None], axis=-1)
            critic = jnp.minimum(critic_1, critic_2) if agent.use_cdq else (critic_1 + critic_2) / 2
            return critic

        compute_Q_DDP = partial(shard_map, mesh=Mesh(devices, ('i',)), in_specs=(P('i'), P('i')), out_specs=(P('i')))(compute_Q)
        observations_repeat = jnp.repeat(jnp.expand_dims(batch['observations'], axis=1), axis=1, repeats=K1 + K2)
        actions_concat = jnp.concatenate([policy_actions, clean_actions], axis=1)
        critic = compute_Q_DDP(actions_concat, observations_repeat)

        critic_policy = critic[:, :K1]
        log_weight_policy = (1 / temp) * critic_policy - jnp.sum((noisy_actions[:, None, :] - (1 - t[..., None]) * policy_actions) ** 2, axis=-1) / (2 * (t ** 2)) - logps_policy_actions
        weight_policy = nn.softmax(log_weight_policy, axis=1)
        vel_estimation_policy = jnp.sum(weight_policy[:, :, None] * ((noisy_actions[:, None, :] - policy_actions) / t[..., None]), axis=1)
        ess_policy = 1.0 / jnp.sum(jnp.square(weight_policy), axis=1)

        critic_clean = critic[:, K1:]
        log_weight_clean = (1 / temp) * critic_clean
        weight_clean = nn.softmax(log_weight_clean, axis=1)
        vel_estimation_clean = jnp.sum(weight_clean[:, :, None] * ((noisy_actions[:, None, :] - clean_actions) / t[..., None]), axis=1)
        ess_clean = 1.0 / jnp.sum(jnp.square(weight_clean), axis=1)

        vel_policy_weight = (ess_policy / (ess_policy + ess_clean + 1e-10))[:, None]
        vel_estimation = vel_policy_weight * vel_estimation_policy + (1.0 - vel_policy_weight) * vel_estimation_clean

        def actor_loss_fn(
                actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            def u_fn(x_t, t, r):
                return agent.actor.apply_fn(
                    {'params': actor_params}, batch['observations'], x_t, t, t - r)
            dt_dt = jnp.ones_like(t)
            dr_dt = jnp.zeros_like(t)
            u_pred, du_dt = jax.jvp(u_fn, (noisy_actions, t, r), (vel_estimation, dt_dt, dr_dt))

            u_tgt = vel_estimation - jnp.clip(t - r, a_min=0.0, a_max=1.0) * du_dt
            assert u_pred.shape == (B, A)
            actor_loss = jnp.power(sg(u_tgt) - u_pred, 2).mean(-1)
            assert actor_loss.shape == (B,)
            metrics = {'actor_loss': actor_loss.mean()}
            return actor_loss.mean(0), metrics

        key, rng = jax.random.split(rng, 2)
        grads, metrics = jax.grad(actor_loss_fn, has_aux=True)(
            agent.actor.params)
        metrics.update({'entropy': -logps_policy_actions.mean()})
        actor = agent.actor.apply_gradients(grads=grads)
        new_agent = agent.replace(
            actor=actor,
            rng=rng)
        return new_agent, metrics, policy_actions[:, 0]

    def update_logp(agent, batch: DatasetDict, policy_actions) -> Tuple[Agent, Dict[str, float]]:
        B, A = batch['actions'].shape
        key, rng = jax.random.split(agent.rng, 2)
        t, r = tr_sampler(key, B, agent.time_dist_name, agent.data_proportion)
        key, rng = jax.random.split(rng, 2)
        noise = jax.random.normal(
            key, (B, agent.act_dim))
        noisy_actions = (1 - t) * policy_actions + t * noise

        N = agent.div_samples_num
        normal_sample = jax.random.normal(
            key, (B, N, agent.act_dim))
        noisy_actions_repeat = jnp.repeat(jnp.expand_dims(noisy_actions, axis=1), axis=1, repeats=N) # [B, N, A]
        observations_repeat = jnp.repeat(jnp.expand_dims(batch['observations'], axis=1), axis=1, repeats=N)
        t_repeat = jnp.repeat(jnp.expand_dims(t, axis=1), axis=1, repeats=N) # [B, N, 1]

        def u_fn(x_t, t, r):
            return agent.actor.apply_fn(
                {'params': agent.actor.params}, observations_repeat, x_t, t, t - r)
        vel, jvp = jax.jvp(u_fn, 
                        (noisy_actions_repeat, t_repeat, t_repeat), 
                        (normal_sample, jnp.zeros_like(t_repeat), jnp.zeros_like(t_repeat)))
        div_estimation = jnp.mean(jnp.sum(jvp * normal_sample, axis=-1), axis=1, keepdims=True)

        def logp_loss_fn(
                logp_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            def logp_u_fn(x_t, t, r):
                return agent.logp_mvel.apply_fn(
                    {'params': logp_params}, batch['observations'], x_t, t, t - r)
            dt_dt = jnp.ones_like(t)
            dr_dt = jnp.zeros_like(t)
            logp_u_pred, du_dt = jax.jvp(logp_u_fn, (noisy_actions, t, r), (vel[:, 0, :], dt_dt, dr_dt))

            logp_u_tgt = div_estimation - jnp.clip(t - r, a_min=0.0, a_max=1.0) * du_dt
            assert logp_u_pred.shape == (B, 1)
            logp_loss = jnp.power(sg(logp_u_tgt) - logp_u_pred, 2).mean(-1)
            assert logp_loss.shape == (B,)
            metrics = {'logp_loss': logp_loss.mean()}
            return logp_loss.mean(0), metrics

        key, rng = jax.random.split(rng, 2)
        grads, metrics = jax.grad(logp_loss_fn, has_aux=True)(
            agent.logp_mvel.params)
        logp_mvel = agent.logp_mvel.apply_gradients(grads=grads)
        new_agent = agent.replace(
            logp_mvel=logp_mvel,
            rng=rng)
        return new_agent, metrics
    
    def update_temperature(self, entropy) -> Tuple[Agent, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info
    

    @jax.jit
    def sample_actions(self, observations: jnp.ndarray):
        return self.eval_actions_sample(observations)
    
    def eval_actions(self, observations: jnp.ndarray):
        if self.eval_action_selection:
            return self.eval_actions_select(observations, self.eval_candidate_num)
        else:
            return self.eval_actions_sample(observations)

    @jax.jit
    def eval_actions_sample(self, observations: jnp.ndarray):
        rng = self.rng
        assert len(observations.shape) == 1
        observations = observations[None]

        key, rng = jax.random.split(rng)
        noise = jax.random.normal(
            key, (1, self.act_dim))
        actions = action_sampler(
            self.actor.apply_fn,
            self.actor.params,
            self.T, noise,
            observations,
            self.clip_sampler)

        assert actions.shape == (1, self.act_dim)
        _, rng = jax.random.split(rng, 2)
        return jnp.squeeze(actions), self.replace(rng=rng)
    
    @partial(jax.jit, static_argnames='cand_num')
    def eval_actions_select(self, observations: jnp.ndarray, cand_num: int = 10):
        rng = self.rng
        assert len(observations.shape) == 1
        observations = observations[None]
        observations = jnp.repeat(observations, repeats=cand_num, axis=0)

        key, rng = jax.random.split(rng)
        noise = jax.random.normal(
            key, (cand_num, self.act_dim))
        actions = action_sampler(
            self.actor.apply_fn,
            self.actor.params,
            self.T, noise,
            observations,
            self.clip_sampler)
        
        q_dist_1 = self.target_critic_1.apply_fn(
            {"params": self.target_critic_1.params}, observations, actions)
        q_dist_2 = self.target_critic_2.apply_fn(
            {"params": self.target_critic_2.params}, observations, actions)
        q_1 = jnp.sum(jax.nn.softmax(q_dist_1, axis=-1) * self.z_atoms[None], axis=-1)
        q_2 = jnp.sum(jax.nn.softmax(q_dist_2, axis=-1) * self.z_atoms[None], axis=-1)
        Q = jnp.minimum(q_1, q_2) if self.use_cdq else (q_1 + q_2) / 2

        actions = actions[jnp.argmax(Q, axis=0)]
        assert actions.shape == (self.act_dim,)
        _, rng = jax.random.split(rng, 2)
        return actions, self.replace(rng=rng)
    
    @jax.jit
    def eval_actions_sample_batch(self, observations: jnp.ndarray):
        rng = self.rng

        key, rng = jax.random.split(rng)
        noise = jax.random.normal(
            key, (observations.shape[0], self.act_dim))
        actions = action_sampler(
            self.actor.apply_fn,
            self.actor.params,
            self.T, noise,
            observations,
            self.clip_sampler)

        _, rng = jax.random.split(rng, 2)
        return actions, self.replace(rng=rng)
    
    
    def calc_target_value(self, observations, actions):
        q_dist_1 = self.target_critic_1.apply_fn(
            {"params": self.target_critic_1.params}, observations, actions)
        q_dist_2 = self.target_critic_2.apply_fn(
            {"params": self.target_critic_2.params}, observations, actions)
        q_1 = jnp.sum(jax.nn.softmax(q_dist_1, axis=-1) * self.z_atoms[None], axis=-1)
        q_2 = jnp.sum(jax.nn.softmax(q_dist_2, axis=-1) * self.z_atoms[None], axis=-1)
        Q = jnp.minimum(q_1, q_2) if self.use_cdq else (q_1 + q_2) / 2

        return Q
    
    def calc_value(self, observations, actions):
        q_dist_1 = self.critic_1.apply_fn(
            {"params": self.critic_1.params}, observations, actions)
        q_dist_2 = self.critic_2.apply_fn(
            {"params": self.critic_2.params}, observations, actions)
        q_1 = jnp.sum(jax.nn.softmax(q_dist_1, axis=-1) * self.z_atoms[None], axis=-1)
        q_2 = jnp.sum(jax.nn.softmax(q_dist_2, axis=-1) * self.z_atoms[None], axis=-1)
        Q = jnp.minimum(q_1, q_2) if self.use_cdq else (q_1 + q_2) / 2

        return Q
    
    def sample_actions_with_logp(self, observations: jnp.ndarray):
        key, rng = jax.random.split(self.rng)
        noise = jax.random.normal(
            key, (observations.shape[0], self.act_dim))
        actions, logp = action_sampler_with_logp(
            self.actor.apply_fn,
            self.actor.params,
            self.logp_mvel.apply_fn,
            self.logp_mvel.params,
            self.T, noise,
            observations,
            self.clip_sampler)

        _, rng = jax.random.split(rng, 2)
        return actions, logp, self.replace(rng=rng)

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int=1):
        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_q(mini_batch)

        true_steps = new_agent.critic_1.step / utd_ratio

        # delayed to update actor, logp and temperature
        new_agent, info = jax.lax.cond(
            (true_steps + 1) % new_agent.actor_delay == 0,
            new_agent.delay_update,
            lambda _: (new_agent, {"actor_loss": 0.0, "entropy": 0.0, "logp_loss": 0.0, "temperature_loss": 0.0, "temperature": 0.0}),
            mini_batch,
        )

        return new_agent, {**critic_info, **info}

    def delay_update(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        new_agent, actor_info, policy_actions = self.update_actor(batch)
        new_agent, logp_info = new_agent.update_logp(batch, policy_actions)
        new_agent, temp_info = new_agent.update_temperature(actor_info['entropy'])
        return new_agent, {**actor_info, **logp_info, **temp_info}
