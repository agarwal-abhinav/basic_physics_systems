from spring_mass.system import DiscreteSpringMassSystem
from spring_mass.controllers import SpringMassLQRController
from spring_mass.animate_spring import plot_spring
import argparse
import os
import numpy as np
import random
import pickle
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate data for the physics system.")
    parser.add_argument('--system_type', type=str, required=True, choices=['spring_mass', 'other_system'], help="Type of system to simulate.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the generated data.")
    parser.add_argument('--num_samples', type=int, default=100, help="Number of samples to generate. Default is 1000.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility. Default is 42.")
    parser.add_argument('--generate_images', action='store_true', help="Whether to generate images for each sample.")
    return parser.parse_args()

def process_system(system_type):
    if system_type == "spring_mass": 
        system = DiscreteSpringMassSystem(m=20, k=40, b=3, dt=0.05, method='zoh')
        controller = SpringMassLQRController(Q=np.eye(2)*10, R=np.eye(1)*0.1, system=system)

        return system, controller
    else:
        raise ValueError(f"Unknown system type: {system_type}")
    
def get_data_for_spring_mass(system, controller, seeds, output_dir): 
    x0_min = -1
    x0_max = 1
    v0_min = -1
    v0_max = 1
    total_time = 25.0 
    burn_in_time = 2.0

    metadata = {
        "system": "spring_mass", 
        "system_params": {
            "m": system.m, 
            "k": system.k, 
            "b": system.b, 
            "dt": system.dt
        },
        "x0_min": x0_min,
        "x0_max": x0_max,
        "v0_min": v0_min,
        "v0_max": v0_max,
        "total_time": total_time,
        "burn_in_time": burn_in_time,
        "num_samples": len(seeds)
    }

    with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f: 
        pickle.dump(metadata, f)

    max_x = -np.inf
    for idx, seed in enumerate(tqdm(seeds)):
        random.seed(seed) 
        np.random.seed(seed)

        x0 = random.uniform(x0_min, x0_max)
        v0 = random.uniform(v0_min, v0_max)

        states, controls, times = system.simulate_with_burn_in(x0, v0, total_time, burn_in_time, controller)

        os.makedirs(os.path.join(output_dir, f"{seed}"), exist_ok=True)

        np.save(os.path.join(output_dir, f"{seed}", "states.npy"), states)
        np.save(os.path.join(output_dir, f"{seed}", "controls.npy"), controls)
        np.save(os.path.join(output_dir, f"{seed}", "times.npy"), times)

        this_max = np.max(np.abs(states[:, 0]))
        if this_max > max_x:
            max_x = this_max

    metadata["max_x"] = max_x
    with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

def main():
    args = parse_arguments()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    system, controller = process_system(args.system_type)

    seeds = list(range(args.seed, args.seed + args.num_samples))

    if args.system_type == "spring_mass": 
        get_data_for_spring_mass(system, controller, seeds, args.output_dir)
        if args.generate_images: 
            with open(os.path.join(args.output_dir, "metadata.pkl"), "rb") as f:
                metadata = pickle.load(f)
            max_x = metadata["max_x"]
            spring_length = 1.5
            while spring_length <= max_x: 
                spring_length += 0.5
            metadata["spring_length"] = spring_length
            metadata['spring_n'] = 30
            metadata['spring_y'] = 4
            metadata['image_dpi'] = 20
            metadata['do_square_images'] = True
            with open(os.path.join(args.output_dir, "metadata.pkl"), "wb") as f: 
                pickle.dump(metadata, f)
            print(f"Using spring length: {spring_length}")
            for seed in tqdm(seeds):
                states = np.load(os.path.join(args.output_dir, f"{seed}", "states.npy"))
                plot_spring(states, os.path.join(args.output_dir, f"{seed}"), 
                            max_x, save_images=True, spring_length=spring_length, 
                            spring_n=metadata['spring_n'], spring_y=metadata['spring_y'], 
                            image_dpi=metadata['image_dpi'], do_square_images=metadata['do_square_images'])
    else: 
        raise ValueError(f"Data generation for system type {args.system_type} not implemented.")

    # Save data to output directory
    print(f"Data saved to {args.output_dir}")

if __name__ == "__main__":
    main()
