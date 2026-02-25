from matplotlib.pylab import *
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patches as patches
π = pi

__all__ = [
    'Truck',
    'deg2rad',
]

# Constants
TRUCK_SPEED = -0.1       # m/s
ENV_X_RANGE = (0, 40)    # m
ENV_Y_RANGE = (-15, 15)  # m

# Train settings
TRAIN_X_CAB_RANGE = (10, 35)                       # m
TRAIN_Y_CAB_RANGE_ABS = (2, 7)                     # m
TRAIN_CAB_ANGLE_RANGE_ABS = (10, 180)              # rad
TRAIN_CAB_TRAILER_ANGLE_DIFF_RANGE_ABS = (10, 45)  # rad
TRAIN_NUM_LESSONS = 10

# Test settings
TEST_X_CAB_RANGE = (10, 35)                    # m
TEST_Y_CAB_RANGE = (-7, 7)                     # m
TEST_CAB_ANGLE_RANGE = (-180, 180)             # rad
TEST_CAB_TRAILER_ANGLE_DIFF_RANGE = (-45, 45)  # rad

class Truck:
    def __init__(self, lesson=10, display=False, gif=False, show_training_region=False):

        self.W = 1 
        self.L = 1 * self.W 
        self.d = 4 * self.L 
        self.s = TRUCK_SPEED
        self.display = display
        self.lesson = lesson
        
        self.box = [ENV_X_RANGE[0], ENV_X_RANGE[1], ENV_Y_RANGE[0], ENV_Y_RANGE[1]]
        self.trailer_trajectory = []
        self.cab_trajectory = []        
        self.frame_num = 0
        self.frames = []
        self.gif = gif
        
        if self.display:
            self.f = figure(figsize=(9, 5), dpi = 100, facecolor='none')#, num='The Truck Backer-Upper')
            self.ax = self.f.add_axes([0.01, 0.01, 0.98, 0.98], facecolor='black')
            self.patches = list()
                
            # self.ax.axis('equal')
            b = self.box
            self.ax.axis([b[0], b[1], b[2], b[3]])
            self.ax.axis('equal')
            # self.ax.set_xlim(right=b[0])
            self.ax.set_xticks([]); self.ax.set_yticks([])
            self.ax.axhline(); self.ax.axvline()

            if show_training_region:
                red_x0 = TRAIN_X_CAB_RANGE[0]
                red_y0 = -TRAIN_Y_CAB_RANGE_ABS[1]
                red_width = TRAIN_X_CAB_RANGE[1] - red_x0
                red_height = TRAIN_Y_CAB_RANGE_ABS[1]*2
                
                rectangle_red = patches.Rectangle((red_x0, red_y0), red_width, red_height,
                                                  edgecolor = "darkred",     
                                                  facecolor = "none",  
                                                  alpha=1,                 
                                                  linewidth=3)
                
                self.ax.add_patch(rectangle_red)   
                                
                plt.text(red_x0 + red_width - 0.1, 
                        red_y0 + red_height - 0.1,
                        'Training Region',
                        color='white',
                        ha='right',
                        va='top',
                        fontsize=5,
                        fontweight='bold')                                    
                                          
                plt.scatter(0, 0, marker='x', color="darkgray", s=60, zorder=10, label = "Target")                                  
            
    def reset(self, ϕ=0, train_test="train", test_seed=1):
        self.trailer_trajectory.clear()
        self.cab_trajectory.clear()
        
        self.ϕ = ϕ 
        if train_test == "train":
            config = _train_configs.get(self.lesson)
        else: 
            config = _test_config
        
        if config is None: 
            raise ValueError(f"No configuration found")       

        if train_test == "test": 
            seed(test_seed)
            
        self.x = uniform(*config["x_range"])
        self.y = uniform(*config["y_range"])                    
        self.θ0 = deg2rad(uniform(*config["θ0_range"]))
        self.θ1 = deg2rad(uniform(*config["Δθ_range"])) + self.θ0
    
        if not self.valid():
            self.reset(ϕ)
        
        if self.display: 
            self.draw()      

    def set(self, x, y, θ0, θ1):
        self.x = x
        self.y = y
        self.θ0 = θ0
        self.θ1 = θ1
        return self.x, self.y, self.θ0, *self._trailer_xy(), self.θ1

    def step(self, ϕ=0, dt=1):
        
        if self.is_jackknifed():
            print('The truck is jackknifed!')
            return
        
        if self.is_offscreen():
            print('The cab or trailer is off screen')
            return
        
        self.ϕ = ϕ
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()
        
        self.x += s * cos(θ0) * dt
        self.y += s * sin(θ0) * dt
        self.θ0 += s / L * tan(ϕ) * dt
        self.θ1 += s / d * sin(θ0 - θ1) * dt

        self.trailer_trajectory.append(self._trailer_xy()) 
        self.cab_trajectory.append((self.x, self.y))
                        
        return (self.x, self.y, self.θ0, *self._trailer_xy(), self.θ1)
    
    def state(self):
        return (self.x, self.y, self.θ0, *self._trailer_xy(), self.θ1)
    
    def update_state(self, state): 
        self.ϕ, self.x, self.y, self.θ0, self.θ1 = state.tolist()    
    
    def _get_atributes(self):
        return (
            self.x, self.y, self.W, self.L, self.d, self.s,
            self.θ0, self.θ1, self.ϕ
        )
    
    def _trailer_xy(self):
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()
        return x - d * cos(θ1), y - d * sin(θ1)
        
    def is_jackknifed(self):
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()   
        abs_diff_deg = abs(rad2deg(θ0 - θ1))
        return min(abs_diff_deg,  abs(abs_diff_deg - 360)) > 90
    
    def is_offscreen(self):
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()
        
        x1, y1 = x + 1.5 * L * cos(θ0), y + 1.5 * L * sin(θ0)
        x2, y2 = self._trailer_xy()
        
        b = self.box
        return not (
            b[0] <= x1 <= b[1] and b[2] <= y1 <= b[3] and
            b[0] <= x2 <= b[1] and b[2] <= y2 <= b[3]
        )
        
    def valid(self):
        return not self.is_jackknifed() and not self.is_offscreen()
        
    def draw(self):
        if not self.display: return
        if self.patches: self.clear()
        self._draw_cab()
        self._draw_trailer()
        self.f.canvas.draw()
        # plt.pause(0.001)
        
        if self.gif:
            buf = BytesIO()
            self.f.savefig(buf, format='png', facecolor='black', dpi=300)
            buf.seek(0)
            image = Image.open(buf).convert("RGBA")  
            self.frames.append(np.array(image))
            buf.close()
            
        self.frame_num += 1
        
    def clear(self):
        for p in self.patches:
            p.remove()
        self.patches = list()
        
    def _draw_cab(self):
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()

        ax = self.ax
        
        x1, y1 = x + L / 2 * cos(θ0), y + L / 2 * sin(θ0)
        bar = Line2D((x, x1), (y, y1), lw=5, color='C2', alpha=1)
        ax.add_line(bar)

        cab = Rectangle(
            (x1, y1 - W / 2), L, W, color='C2', alpha=1, transform=
            matplotlib.transforms.Affine2D().rotate_deg_around(x1, y1, rad2deg(θ0)) +
            ax.transData
        )

        ax.add_patch(cab)

        x2, y2 = x1 + L / 2 ** 0.5 * cos(θ0 + π / 4), y1 + L / 2 ** 0.5 * sin(θ0 + π / 4)
        left_wheel = Line2D(
            (x2 - L / 4 * cos(θ0 + ϕ), x2 + L / 4 * cos(θ0 + ϕ)),
            (y2 - L / 4 * sin(θ0 + ϕ), y2 + L / 4 * sin(θ0 + ϕ)),
            lw=3, color='C5', alpha=1)
        ax.add_line(left_wheel)

        x3, y3 = x1 + L / 2 ** 0.5 * cos(π / 4 - θ0), y1 - L / 2 ** 0.5 * sin(π / 4 - θ0)
        right_wheel = Line2D(
            (x3 - L / 4 * cos(θ0 + ϕ), x3 + L / 4 * cos(θ0 + ϕ)),
            (y3 - L / 4 * sin(θ0 + ϕ), y3 + L / 4 * sin(θ0 + ϕ)),
            lw=3, color='C5', alpha=1)
        ax.add_line(right_wheel)
        
        self.patches += [cab, bar, left_wheel, right_wheel]
        
    def _draw_trailer(self):
        x, y, W, L, d, s, θ0, θ1, ϕ = self._get_atributes()    
        ax = self.ax
             
        x, y = x - d * cos(θ1), y - d * sin(θ1) - W / 2
        trailer = Rectangle(
            (x, y), d, W, color='C0', alpha=1, transform=
            matplotlib.transforms.Affine2D().rotate_deg_around(x, y + W / 2, rad2deg(θ1)) + 
            ax.transData
        )

        ax.add_patch(trailer)
        self.patches += [trailer]

    def _draw_trajectories(self, test_seed): 
        
        trailer_color = '#1f77b4'  
        cab_color = '#ff7f0e'              
                        
        x_trailer_trajectory = [point[0] for point in self.trailer_trajectory]
        y_trailer_trajectory = [point[1] for point in self.trailer_trajectory]
        
        x_cab_trajectory = [point[0] for point in self.cab_trajectory]
        y_cab_trajectory = [point[1] for point in self.cab_trajectory]
        
        # red_x0 = train_x_cab_range[0]
        # red_y0 = -train_y_cab_range_abs[1]
        # red_width = train_x_cab_range[1] - red_x0
        # red_height = train_y_cab_range_abs[1]*2
        
        green_x0 = self.box[0]
        green_y0 = self.box[2]
        green_width = self.box[1] - self.box[0]
        green_height = self.box[3] - self.box[2]                 

        # rectangle_red = patches.Rectangle((red_x0, red_y0), red_width, red_height,
        #                                   facecolor='red',           
        #                                   edgecolor='darkred',       
        #                                   alpha=0.3,                 
        #                                   linewidth=2)          
                
        # rectangle_green = patches.Rectangle((green_x0, green_y0), green_width, green_height,
        #                                     facecolor='green',           
        #                                     edgecolor='darkgreen',       
        #                                     alpha=0.3,                 
        #                                     linewidth=2)                     
        
        plt.figure(figsize=(7.5, 3), dpi=100) 
        
        plt.plot(x_trailer_trajectory, y_trailer_trajectory, 
                 color=trailer_color, linestyle='-', linewidth=1.5, alpha=0.8)
        
        plt.plot(x_cab_trajectory, y_cab_trajectory, 
                 color=cab_color, linestyle='-', linewidth=1.5, alpha=0.8)
        
        plt.scatter(x_trailer_trajectory, y_trailer_trajectory, 
                    color=trailer_color, marker='.', s=15, alpha=0.6)
        
        plt.scatter(x_cab_trajectory, y_cab_trajectory, 
                    color=cab_color, marker='.', s=15, alpha=0.6)
        
        plt.scatter(x_trailer_trajectory[0], y_trailer_trajectory[0], 
                    marker='o', color=trailer_color, s=60, zorder=10, 
                    label='Trailer Start Position')
        
        plt.scatter(x_cab_trajectory[0], y_cab_trajectory[0], 
                    marker='o', color=cab_color, s=60, zorder=10,
                    label='Cab Start Position')
                
        plt.scatter(x_trailer_trajectory[-1], y_trailer_trajectory[-1], 
                   marker='x', color=trailer_color, s=60, zorder=10,
                   label='Trailer End Position')
        
        plt.scatter(x_cab_trajectory[-1], y_cab_trajectory[-1], 
                   marker='x', color=cab_color, s=60, zorder=10,
                   label='Cab End Position')
        
        plt.plot([x_trailer_trajectory[0], x_cab_trajectory[0]], 
                 [y_trailer_trajectory[0], y_cab_trajectory[0]], 
                 'w--', linewidth=1.5) 

        plt.plot([x_trailer_trajectory[-1], x_cab_trajectory[-1]], 
                 [y_trailer_trajectory[-1], y_cab_trajectory[-1]], 
                 'w--', linewidth=1.5)
        
        # plt.gca().add_patch(rectangle_red) 
        # plt.gca().add_patch(rectangle_green)
        
        # plt.text(red_x0 + red_width - 0.1, 
        #          red_y0 + red_height - 0.1,
        #          'Training Region',
        #          color='white',
        #          ha='right',
        #          va='top',
        #          fontsize=5,
        #          fontweight='bold')    

        plt.scatter(0, 0, marker='x', color="darkgray", s=60, zorder=10, label = "Target") 
        plt.tight_layout()
        plt.subplots_adjust(right=0.78)
        plt.xticks([])
        plt.yticks([])        
        plt.xlim(self.box[0], self.box[1])
        plt.ylim(self.box[2], self.box[3])  
        plt.grid(False)      
        
        # directory = f'trajectories/lesson-{self.lesson}-{current_time}'
        
        # if not os.path.exists(directory):
        #     os.makedirs(directory)  
        
        # trajectory_path = f'{directory}/trajectory-{test_seed}.png'
        
        # plt.savefig(trajectory_path, dpi=300, facecolor='white', bbox_inches='tight')  
        # plt.close()    

    def generate_gif(self):
        gif_path = f'./gifs/lesson-{self.lesson}-{current_time}.gif'
        with imageio.get_writer(gif_path, mode='I', fps=50, loop=0) as writer:
            for frame_array in self.frames[::2]:
                writer.append_data(frame_array)
        
        optimized_path = gif_path.replace(".gif", "-optimized.gif")
        subprocess.run(["gifsicle", "-O3", "--colors", "256", gif_path, "-o", optimized_path], check=True)
        os.replace(optimized_path, gif_path)

def _create_train_configs(x_cab_range, y_cab_range_abs, cab_angle_range_abs, cab_trailer_angle_diff_range_abs, num_lessons):
    num_lessons -= 1
    configs = {}
    x_cab_first, x_cab_final = x_cab_range
    y_cab_first, y_cab_final = y_cab_range_abs
    cab_angle_first, cab_angle_final = cab_angle_range_abs
    cab_trailer_angle_diff_first, cab_trailer_angle_diff_final = cab_trailer_angle_diff_range_abs
    x_lower = x_cab_first

    for i in range(1, num_lessons + 1):
        x_upper = x_cab_first + (x_cab_final - x_cab_first) * (i - 1) / (num_lessons - 1)
        y_upper = y_cab_first + (y_cab_final - y_cab_first) * (i - 1) / (num_lessons - 1)        
        θ0_upper = cab_angle_first + (cab_angle_final - cab_angle_first) * (i - 1) / (num_lessons - 1)
        Δθ_upper = cab_trailer_angle_diff_first + (cab_trailer_angle_diff_final - cab_trailer_angle_diff_first) * (i - 1) / (num_lessons - 1)
        
        configs[i] = {
            "x_range": (x_lower, x_upper),
            "y_range": (-y_upper, y_upper),
            "θ0_range": (-θ0_upper, θ0_upper),
            "Δθ_range": (-Δθ_upper, Δθ_upper)
        }
        
        x_lower = x_upper
    
    configs[num_lessons + 1] = {
        "x_range": (x_cab_first, x_upper),
        "y_range": (-y_upper, y_upper),
        "θ0_range": (-θ0_upper, θ0_upper),
        "Δθ_range": (-Δθ_upper, Δθ_upper)
    }
                    
    return configs

_train_configs = _create_train_configs(
    TRAIN_X_CAB_RANGE, 
    TRAIN_Y_CAB_RANGE_ABS, 
    TRAIN_CAB_ANGLE_RANGE_ABS,
    TRAIN_CAB_TRAILER_ANGLE_DIFF_RANGE_ABS, 
    TRAIN_NUM_LESSONS
)

_test_config = {
    "x_range": TEST_X_CAB_RANGE,
    "y_range": TEST_Y_CAB_RANGE,
    "θ0_range": TEST_CAB_ANGLE_RANGE,
    "Δθ_range": TEST_CAB_TRAILER_ANGLE_DIFF_RANGE
}
