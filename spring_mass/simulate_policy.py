from spring_mass.controllers import SpringMassDiffusionController
from spring_mass.system import DiscreteSpringMassSystem
import numpy as np 
import random

def evaluate_policy(system: DiscreteSpringMassSystem, 
                    policy_ckpt: str, 
                    Q, R, x0_min, x0_max, v0_min, v0_max, total_time, 
                    burn_time, min_seed: int = 0, 
                    total_seeds: int = 5, 
                    use_position_only: bool = True, 
                    policy_device: str = 'cpu'): 
    controller = SpringMassDiffusionController(policy_ckpt, use_position_only=use_position_only, policy_device=policy_device)

    costs = []

    for seed in range(min_seed, min_seed + total_seeds): 
        np.random.seed(seed)
        random.seed(seed)
        x0 = random.uniform(x0_min, x0_max)
        v0 = random.uniform(v0_min, v0_max)

        state_traj, u_traj, t_traj = system.simulate_with_burn_in_diffusion(
            x0, v0, total_time, burn_in_time=burn_time, controller=controller
        )

        state_cost = np.einsum('...ij,jk,...ki->...', state_traj.transpose(0, 2, 1), Q, state_traj)
        control_cost = np.einsum('...ij,jk,...ki->...', u_traj.transpose(0, 2, 1), R, u_traj)

        costs.append(state_cost + control_cost)

    return costs
