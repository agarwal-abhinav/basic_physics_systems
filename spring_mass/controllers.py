from email import policy
import numpy as np
import control as ct
import torch 
import hydra
from omegaconf import OmegaConf
import dill
import os
from collections import deque

def load_policy(policy_name: str, dataset_zarr: str = None, load_normalzer_from_file: bool = False): 
    # return
    payload = torch.load(open(policy_name, "rb"), pickle_module=dill)

    model_cfg = payload["cfg"]

    # if INFER_FROZEN_POLICY: 
    #     OmegaConf.set_struct(model_cfg.policy, False)
    #     model_cfg.policy.inference_loading = True
    #     OmegaConf.set_struct(model_cfg.policy, True)

    model_workspace_cls = hydra.utils.get_class(model_cfg._target_)
    model_workspace = model_workspace_cls(model_cfg)
    model_workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    if load_normalzer_from_file: 
        normalizer_path = os.path.join(os.path.dirname(os.path.dirname(policy_name)), "normalizer.pt")
        normalizer = torch.load(normalizer_path, weights_only=False)
    else:
        if dataset_zarr is not None: 
            model_cfg.task.dataset.zarr_path = dataset_zarr
        dataset = hydra.utils.instantiate(model_cfg.task.dataset)
        normalizer = dataset.get_normalizer()

    policy = model_workspace.model 
    policy.set_normalizer(normalizer)
    if model_cfg.training.use_ema: 
        policy = model_workspace.ema_model 
    policy.set_normalizer(normalizer)

    policy.eval()
    return policy 

class SpringMassLinearController: 
    def __init__(self, K): 
        self.K = K 
        
    def calculate(self, state): 
        u = -self.K @ state
        return u

class SpringMassConstantController: 
    def __init__(self, u_const): 
        self.u_const = np.array([[u_const]])
        
    def calculate(self, state): 
        return self.u_const
    
class SpringMassLQRController(SpringMassLinearController): 
    def __init__(self, Q, R, system): 
        Co = ct.ctrb(system.A, system.B) 
        assert np.linalg.matrix_rank(Co) == system.A.shape[0], "System not controllable"

        K, S, E = ct.dlqr(system.A, system.B, Q, R)

        super().__init__(K)

class SpringMassDiffusionController(): 
    def __init__(self, checkpoint_path, use_position_only=False):
        self.policy = load_policy(checkpoint_path, load_normalzer_from_file=True)

        self._obs_deque = deque(maxlen=self.policy.n_obs_steps)
        self._action_deque = deque(maxlen=self.policy.n_obs_steps)

        self._predicted_actions = deque(maxlen=self.policy.n_action_steps)

        self.use_position_only = use_position_only

    def update_obs(self, obs, action): 
        if self.use_position_only: 
            self._obs_deque.append(obs[0:1])
        else: 
            self._obs_deque.append(obs)

        self._action_deque.append(action)

    def calculate(self, state, action): 
        self.update_obs(state, action)

        if len(self._predicted_actions) == 0: 
            # print(self._obs_deque[0].shape)
            this_obs_dict = {
                self.policy.obs_key: torch.tensor(np.concatenate(self._obs_deque, axis=1).T, device=self.policy.device).float().unsqueeze(0), 
                self.policy.action_key: torch.tensor(np.concatenate(self._action_deque, axis=1).T, device=self.policy.device).float().unsqueeze(0)
            }
            # print(this_obs_dict[self.policy.obs_key].shape)

            with torch.no_grad(): 
                pred_control = self.policy.predict_action(this_obs_dict)['action'].cpu().numpy().reshape(-1,)

            # print(pred_control)
            self._predicted_actions.append(np.expand_dims(pred_control, axis=0))

        return self._predicted_actions.popleft()
