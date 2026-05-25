import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.model_cls = "MeanFlowLearner"
    config.actor_lr = 3e-4
    config.logp_lr = 3e-4 # learning rate for average divergence network
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4  # learning rate for temperature
    config.discount = 0.99
    config.tau = 0.005  # for soft target updates.
    config.T = 2  # sampling steps for MeanFlow model
    config.critic_hidden_dims=(256,256,256)
    config.actor_hidden_dims=(256,256,256)
    config.actor_layer_norm = True
    config.critic_layer_norm = True
    config.temp = 0.01  # initial temperature coefficient
    config.backup_entropy = True  # backup entropy when computing Q
    config.vel1_samples_num = 16  # sample number of the policy proposal
    config.vel2_samples_num = 32  # sample number of the Gaussian proposal
    config.eval_action_selection = True  # use action section when testing
    config.eval_candidate_num = 10  # number of action candidates
    config.div_samples_num = 2  # sample number for divergence estimation
    config.time_dist_name = 'logit_normal' # time distribution of MeanFlow model
    config.data_proportion = 0.75  # proportion of setting r = t
    config.target_entropy_coeff = -0.5  # coefficient for target entropy
    config.actor_delay = 1  # delay for actor, average divergence, temperature updates
    config.use_cdq = True  # use clipped double Q-learning
    config.v_min = -1600  # min value for value distribution
    config.v_max = 1600  # max value for value distribution
    config.atoms_num = 101 # number of atoms for value distribution
    return config
