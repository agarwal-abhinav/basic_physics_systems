import numpy as np 
import control as ct
from spring_mass.controllers import SpringMassLQRController, SpringMassConstantController
import pickle
from tqdm import tqdm

class DiscreteSpringMassSystem: 
    def __init__(self, m, k, b, dt, method='euler'): 
        self.m = m
        self.k = k
        self.b = b
        self.dt = dt 

        if method == 'euler': 
            self.A = np.array([[1, dt], 
                            [-dt*(k/m), 1 - dt*(b/m)]])
            self.B = np.array([[0], 
                                [dt/m]])
        elif method == 'zoh': 
            Ac = np.array([[0, 1], 
                           [-(k/m), -(b/m)]])
            Bc = np.array([[0], 
                           [1/m]])
            Cc = np.eye(2) 
            Dc = np.zeros((2, 1))

            sysc = ct.ss(Ac, Bc, Cc, Dc)
            sysd = ct.c2d(sysc, dt, method='zoh')
            self.A = sysd.A
            self.B = sysd.B
        else: 
            raise ValueError("Unknown discretization method")
        
    def reset(self, x0, v0): 
        self.state = np.array([[x0], 
                               [v0]])

    def step(self, u):
        self.state = self.A @ self.state + self.B @ u
        return self.state 
    
    def simulate(self, x0, v0, total_time, controller=None): 
        self.reset(x0, v0)
        states = [self.state.copy()]
        controls = []
        times = [0]

        n_steps = int(total_time / self.dt)

        for t in range(n_steps): 
            if controller is None: 
                u = np.array([[0]])
            else:
                u = controller.calculate(self.state)
            _ = self.step(u)
            states.append(self.state.copy())
            times.append((t+1)*self.dt)
            controls.append(u.copy())
        
        controls.append(controls[-1])  # To make controls same length as states
        return np.array(states), np.array(controls), np.array(times)
    
    def simulate_with_burn_in(self, x0, v0, total_time, burn_in_time, controller=None): 
        burn_in_controller = SpringMassConstantController(0)

        self.reset(x0, v0)

        states = [self.state.copy()]
        controls = []
        times = [0]

        n_steps = int(total_time / self.dt)
        burn_in_steps = int(burn_in_time / self.dt)

        for t in range(n_steps): 
            if t < burn_in_steps: 
                u = burn_in_controller.calculate(self.state)
            else: 
                if controller is None: 
                    u = np.array([[0]])
                else:
                    u = controller.calculate(self.state)
            _ = self.step(u)
            states.append(self.state.copy())
            times.append((t+1)*self.dt)
            controls.append(u.copy())

        controls.append(controls[-1])  # To make controls same length as states
        return np.array(states), np.array(controls), np.array(times)
    
    def simulate_with_burn_in_diffusion(self, x0, v0, total_time, burn_in_time, controller=None): 
        burn_in_controller = SpringMassConstantController(0)

        self.reset(x0, v0)

        states = [self.state.copy()]
        controls = []
        times = [0]

        n_steps = int(total_time / self.dt)
        burn_in_steps = int(burn_in_time / self.dt)

        for t in tqdm(range(n_steps)): 
            if t < burn_in_steps: 
                u = burn_in_controller.calculate(self.state)
                controller.update_obs(self.state, u)
            else: 
                if controller is None: 
                    u = np.array([[0]])
                else:
                    u = controller.calculate(self.state, u if t == burn_in_steps else controls[-1])
            _ = self.step(u)
            states.append(self.state.copy())
            times.append((t+1)*self.dt)
            controls.append(u.copy())

        controls.append(controls[-1])  # To make controls same length as states
        return np.array(states), np.array(controls), np.array(times)
    
    @classmethod
    def create_from_file(cls, file_path):
        """Create an instance of the class from a pickle file."""
        with open(file_path, 'rb') as file:
            attributes = pickle.load(file)
        return cls(**attributes)
