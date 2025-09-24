from tqdm import tqdm 
import os 
import zarr 
import numpy as np
from PIL import Image
import pickle 

def create_zarr(directory, zarr_path, state_file_name='states.npy', action_file_name='controls.npy', 
                observation_file_name=None, max_traj=15000, use_only_position=True, context_length=None): 
    # zarr creation code 
    existing_dirs = [int(d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and d.isdigit()]

    concatenated_states = []
    concatenated_observations = []
    concatenated_actions = []
    episode_ends = []
    current_end = 0

    if observation_file_name is None: 
        observation_file_name = state_file_name

    if context_length is not None: 
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
                metadata = pickle.load(f)
        burn_in_time = metadata['burn_in_time']
        dt = metadata['system_params']['dt']
        burn_in_steps = int(burn_in_time / dt)
        start_index = burn_in_steps - context_length
        assert start_index >= 0, "Context length too long for burn-in time"

        zarr_path = zarr_path.replace(".zarr", f"_obs_{context_length}.zarr")
    else:
        start_index = 0

    m = 0
    for this_dir in tqdm(existing_dirs): 
        state_file = os.path.join(directory, str(this_dir), state_file_name)
        observation_file = os.path.join(directory, str(this_dir), observation_file_name)
        action_file = os.path.join(directory, str(this_dir), action_file_name)

        this_state = np.load(state_file)
        if len(this_state.shape) == 3: 
            this_state = this_state.squeeze(-1)[start_index:, :]
        this_observation = np.load(observation_file)
        if len(this_observation.shape) == 3: 
            this_observation = this_observation.squeeze(-1)[start_index:, :]
        if use_only_position: 
            this_observation = this_observation[:, 0:1]
        this_action = np.load(action_file)
        if len(this_action.shape) == 3: 
            this_action = this_action.squeeze(-1)[start_index:, :]

        concatenated_states.append(this_state)
        concatenated_observations.append(this_observation)
        concatenated_actions.append(this_action)
        episode_ends.append(current_end + len(this_state))
        current_end += len(this_state)

        m += 1
        if m >= max_traj: 
            break

    root = zarr.open_group(zarr_path, mode='w')
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    # Chunk sizes optimized for read (not for supercloud storage, sorry admins)
    state_chunk_size = (1024, this_state.shape[1])
    action_chunk_size = (2048, this_action.shape[1])
    observation_chunk_size = (1024, this_observation.shape[1])

    # convert to numpy
    concatenated_states = np.concatenate(concatenated_states, axis=0)
    concatenated_actions = np.concatenate(concatenated_actions, axis=0)
    concatenated_observations = np.concatenate(concatenated_observations, axis=0)
    episode_ends = np.array(episode_ends)

    assert episode_ends[-1] == concatenated_states.shape[0]
    assert concatenated_states.shape[0] == concatenated_actions.shape[0]
    assert concatenated_states.shape[0] == concatenated_observations.shape[0]

    data_group.create_dataset("state", data=concatenated_states, chunks=state_chunk_size)
    data_group.create_dataset("action", data=concatenated_actions, chunks=action_chunk_size)
    data_group.create_dataset("observation", data=concatenated_observations, chunks=observation_chunk_size)
    meta_group.create_dataset("episode_ends", data=episode_ends)

def create_zarr_with_images(directory, zarr_path, state_file_name='states.npy', action_file_name='controls.npy', 
                observation_file_name=None, max_traj=15000, use_only_position=True, context_length=None): 
    # zarr creation code 
    existing_dirs = [int(d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and d.isdigit()]

    concatenated_states = []
    concatenated_observations = []
    concatenated_actions = []
    concatenated_images = []
    concatenated_targets = []
    episode_ends = []
    current_end = 0

    if observation_file_name is None: 
        observation_file_name = state_file_name

    if context_length is not None: 
        with open(os.path.join(directory, "metadata.pkl"), "rb") as f:
                metadata = pickle.load(f)
        burn_in_time = metadata['burn_in_time']
        dt = metadata['system_params']['dt']
        burn_in_steps = int(burn_in_time / dt)
        start_index = burn_in_steps - context_length
        assert start_index >= 0, "Context length too long for burn-in time"

        zarr_path = zarr_path.replace(".zarr", f"_obs_{context_length}.zarr")
    else:
        start_index = 0

    m = 0
    for this_dir in tqdm(existing_dirs): 
        state_file = os.path.join(directory, str(this_dir), state_file_name)
        observation_file = os.path.join(directory, str(this_dir), observation_file_name)
        action_file = os.path.join(directory, str(this_dir), action_file_name)

        this_state = np.load(state_file)
        if len(this_state.shape) == 3: 
            this_state = this_state.squeeze(-1)[start_index:, :]
        this_observation = np.load(observation_file)
        if len(this_observation.shape) == 3: 
            this_observation = this_observation.squeeze(-1)[start_index:, :]
        if use_only_position: 
            this_observation = this_observation[:, 0:1]
        this_action = np.load(action_file)
        if len(this_action.shape) == 3: 
            this_action = this_action.squeeze(-1)[start_index:, :]

        this_images = []
        for j in range(start_index, start_index + this_state.shape[0]):
            image_file = os.path.join(directory, str(this_dir), f"frame_{j:04d}.png")
            this_image = Image.open(image_file)
            this_image = this_image.convert('RGB')
            this_image = this_image.resize((64, 64))
            this_image_numpy = np.array(this_image)
            this_images.append(this_image_numpy)

        this_images = np.array(this_images)

        concatenated_images.append(this_images)
        concatenated_states.append(this_state)
        concatenated_observations.append(this_observation)
        concatenated_targets.append(np.zeros_like(this_observation))  # Placeholder for target, can be modified as needed
        concatenated_actions.append(this_action)
        episode_ends.append(current_end + len(this_state))
        current_end += len(this_state)

        m += 1
        if m >= max_traj: 
            break

    root = zarr.open_group(zarr_path, mode='w')
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    # Chunk sizes optimized for read (not for supercloud storage, sorry admins)
    state_chunk_size = (1024, this_state.shape[1])
    action_chunk_size = (2048, this_action.shape[1])
    observation_chunk_size = (1024, this_observation.shape[1])
    image_chunk_size = (1024, *this_images[0].shape)

    # convert to numpy
    concatenated_states = np.concatenate(concatenated_states, axis=0)
    concatenated_actions = np.concatenate(concatenated_actions, axis=0)
    concatenated_observations = np.concatenate(concatenated_observations, axis=0)
    concatenated_targets = np.concatenate(concatenated_targets, axis=0)
    concatenated_images = np.concatenate(concatenated_images, axis=0)
    episode_ends = np.array(episode_ends)

    assert episode_ends[-1] == concatenated_states.shape[0]
    assert concatenated_states.shape[0] == concatenated_actions.shape[0]
    assert concatenated_states.shape[0] == concatenated_observations.shape[0]
    assert concatenated_states.shape[0] == concatenated_images.shape[0]

    data_group.create_dataset("state", data=concatenated_states, chunks=state_chunk_size)
    data_group.create_dataset("action", data=concatenated_actions, chunks=action_chunk_size)
    data_group.create_dataset("observation", data=concatenated_observations, chunks=observation_chunk_size)
    data_group.create_dataset("target", data=concatenated_targets, chunks=observation_chunk_size)
    data_group.create_dataset("image", data=concatenated_images, chunks=image_chunk_size)
    meta_group.create_dataset("episode_ends", data=episode_ends)

if __name__ == "__main__": 
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, required=True, help="Directory containing trajectory subdirectories")
    parser.add_argument("--zarr_path", type=str, required=True, help="Path to save the zarr file")
    parser.add_argument("--state_file_name", type=str, default="states.npy", help="Name of the state file in each trajectory directory")
    parser.add_argument("--action_file_name", type=str, default="controls.npy", help="Name of the action file in each trajectory directory")
    parser.add_argument("--observation_file_name", type=str, default=None, help="Name of the observation file in each trajectory directory (if different from state)")
    parser.add_argument("--max_traj", type=int, default=15000, help="Maximum number of trajectories to process")
    parser.add_argument("--use_only_position", action='store_true', help="Use only position from state as observation")
    parser.add_argument("--use_images", action='store_true', help="Whether to generate images for each sample")
    parser.add_argument("--context_length", type=int, default=None, help="If specified, use this context length from burn-in period as observation")
    args = parser.parse_args()

    if args.use_images: 
        create_zarr_with_images(args.directory, args.zarr_path, state_file_name=args.state_file_name, 
                action_file_name=args.action_file_name, observation_file_name=args.observation_file_name,
                max_traj=args.max_traj, use_only_position=args.use_only_position, context_length=args.context_length)
    else:
        create_zarr(args.directory, args.zarr_path, state_file_name=args.state_file_name, 
                    action_file_name=args.action_file_name, observation_file_name=args.observation_file_name,
                    max_traj=args.max_traj, use_only_position=args.use_only_position, context_length=args.context_length)