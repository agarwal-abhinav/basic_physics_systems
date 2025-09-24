import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
from matplotlib.patches import Circle
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

def generate_spring(n, spring_y=4): 
    data = np.zeros((2, n+2))
    data[:, -1] = [1, 0]

    for i in range(1, n+1): 
        data[0, i] = (2*i-1)/(2*n)
        data[1, i] = -spring_y/(2*n) if i % 2 else spring_y/(2*n)
     
    return data 

def plot_spring(states, save_dir, max_x, spring_n=30, fig=None, 
                spring_length=4.0, save_images=False, spring_y=4, image_dpi=30, 
                do_square_images=True):
    if fig is None: 
        fig = plt.figure()
        ax = fig.add_subplot(aspect='equal')
        ax.set_xlim(0, spring_length + max_x + 0.5)
        if do_square_images: 
            ax.set_ylim(- (spring_length + max_x + 0.5)/2, (spring_length + max_x + 0.5)/2)
        else:
            ax.set_ylim(-1.5, 1.5)
    
    xs = states[:, 0]

    assert spring_length > max_x, "Spring length must be greater than max x displacement"

    data = np.append(generate_spring(spring_n, spring_y=spring_y), np.ones((1, spring_n+2)), axis=0)

    x0 = xs[0,0] + spring_length 
    spring = Line2D(data[0, :], data[1, :], color='r')
    circle = ax.add_patch(Circle((x0, 0), 0.25, fc='b', zorder=3))
    ax.add_line(spring)

    def animate(i): 
        x = xs[i, -0] + spring_length
        circle.set_center((x, 0))

        stretch_factor = x 

        A = Affine2D().scale(stretch_factor, 8/stretch_factor).get_matrix()
        data_new = np.matmul(A, data)

        xn = data_new[0, :]
        yn = data_new[1, :] 

        spring.set_data(xn, yn)

        plt.axis('off')

        if save_images: 
            plt.axis('off')
            plt.savefig(f'{save_dir}/frame_{i:04d}.png', bbox_inches='tight', dpi=image_dpi)

    ani = animation.FuncAnimation(fig, animate, frames=len(xs))
    # ffmpeg_writer = animation.FFMpegWriter(fps=30)
    # ani.save(f'{save_dir}/spring_mass.gif', writer=ffmpeg_writer)
    ani.save(f'{save_dir}/animation.gif', writer=PillowWriter(fps=30), dpi=image_dpi, savefig_kwargs={'pad_inches': 0})

    plt.close('all')






