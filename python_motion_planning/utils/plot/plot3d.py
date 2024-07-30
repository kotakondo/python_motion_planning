"""
Plot tools 2D
@author: huiming zhou
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..environment.env3d import Env3D, Grid3D, Node3D

class Plot3D:
    def __init__(self, start, goal, env: Env3D):
        self.start = Node3D(start, start, 0, 0)
        self.goal = Node3D(goal, goal, 0, 0)
        self.env = env
        size = 10
        self.fig = plt.figure("planning", figsize=(size, size))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def animation(self, path: list, name: str, cost: float = None, expand: list = None, history_pose: list = None,
                  predict_path: list = None, lookahead_pts: list = None, cost_curve: list = None,
                  ellipse: np.ndarray = None) -> None:
        name = name + "\ncost: " + str(cost) if cost else name
        self.plotEnv(name)
        if expand is not None:
            self.plotExpand(expand)
        if history_pose is not None:
            self.plotHistoryPose(history_pose, predict_path, lookahead_pts)
        if path is not None:
            self.plotPath(path)

        if cost_curve:
            plt.figure("cost curve")
            self.plotCostCurve(cost_curve, name)

        if ellipse is not None:
            self.plotEllipse(ellipse)

        plt.show()
    
    def dynamic_animation(self, paths: list[list], name: str, cost: float = None, expand: list = None, colors: list = None, history_pose: list = None,
                  predict_path: list = None, lookahead_pts: list = None, cost_curve: list = None,
                  ellipse: np.ndarray = None, fig_name: str = None) -> None:

        name = name + "\ncost: " + str(cost) if cost else name

        for path_idx, path in enumerate(paths):
            
            self.plotEnv(name, path_idx, colors[path_idx]) if path_idx == len(paths) - 1 else None
            
            if expand is not None:
                self.plotExpand(expand)
            if history_pose is not None:
                self.plotHistoryPose(history_pose, predict_path, lookahead_pts)
            if path is not None:
                self.plotPath(path, path_color=colors[path_idx])

            if cost_curve:
                plt.figure("cost curve")
                self.plotCostCurve(cost_curve, name)

            if ellipse is not None:
                self.plotEllipse(ellipse)

        plt.show()
        # save fig to png
        plt.savefig(f"/home/kkondo/Downloads/tmp/{fig_name if fig_name else 'tmp.png'}")
        plt.close()

    def plotCurrentDynamicEnv(self, name: str, color="black") -> None:
        '''
        Plot environment with static and dynamic obstacles in 3D.

        Parameters
        ----------
        name: Algorithm name or some other information
        interval_idx: Optional index for intervals
        color: Color for dynamic obstacles
        '''

        # Plot start and goal points
        self.ax.scatter(self.start.x, self.start.y, self.start.z, marker="s", color="#ff0000", label="Start")
        self.ax.scatter(self.goal.x, self.goal.y, self.goal.z, marker="s", color="#1155cc", label="Goal")

        if isinstance(self.env, Grid3D):
            
            # Static obstacles
            obs_x = [x[0] for x in self.env.static_obstacles]
            obs_y = [x[1] for x in self.env.static_obstacles]
            obs_z = [x[2] for x in self.env.static_obstacles]
            # self.ax.scatter(obs_x, obs_y, obs_z, s=100, color='black', marker='s', alpha=0.2, label='Static Obstacles')

            # Dynamic obstacles
            obs_x = [x[0] for x in self.env.dynamic_obstacles]
            obs_y = [x[1] for x in self.env.dynamic_obstacles]
            obs_z = [x[2] for x in self.env.dynamic_obstacles]
            self.ax.scatter(obs_x, obs_y, obs_z, s=100, color=color, marker='s', alpha=0.2, label='Dynamic Obstacles')

        self.ax.set_title(name)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        
        # set axis limits
        self.ax.set_xlim(0, self.env.x_range)
        self.ax.set_ylim(0, self.env.y_range)
        self.ax.set_zlim(0, self.env.z_range)
        plt.show()

    def plotEnv(self, name: str, interval_idx: int = 0, color="black") -> None:
        '''
        Plot environment with static and dynamic obstacles in 3D.

        Parameters
        ----------
        name: Algorithm name or some other information
        interval_idx: Optional index for intervals
        color: Color for dynamic obstacles
        '''

        # Plot start and goal points
        self.ax.scatter(self.start.x, self.start.y, self.start.z, marker="s", color="#ff0000", label="Start")
        self.ax.scatter(self.goal.x, self.goal.y, self.goal.z, marker="s", color="#1155cc", label="Goal")

        if isinstance(self.env, Grid3D):
            
            self.ax.set_title(name + "\ninterval index: " + str(interval_idx))
            dynamic_obstacles, static_obstacles = self.env.get_obstacles_from_interval_index_for_plot(interval_idx)
            
            # Static obstacles
            obs_x = [x[0] for x in static_obstacles]
            obs_y = [x[1] for x in static_obstacles]
            obs_z = [x[2] for x in static_obstacles]
            self.ax.scatter(obs_x, obs_y, obs_z, s=100, color='black', marker='s', alpha=0.2, label='Static Obstacles')

            # Dynamic obstacles
            obs_x = [x[0] for x in dynamic_obstacles]
            obs_y = [x[1] for x in dynamic_obstacles]
            obs_z = [x[2] for x in dynamic_obstacles]
            self.ax.scatter(obs_x, obs_y, obs_z, s=100, color=color, marker='s', alpha=0.2, label='Dynamic Obstacles')

        self.ax.set_title(name)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        
        # set axis limits
        self.ax.set_xlim(0, self.env.x_range)
        self.ax.set_ylim(0, self.env.y_range)
        self.ax.set_zlim(0, self.env.z_range)

    def plotExpand(self, expand: list) -> None:
        '''
        Plot expanded grids using in graph searching.

        Parameters
        ----------
        expand: Expanded grids during searching
        '''
        if self.start in expand:
            expand.remove(self.start)
        if self.goal in expand:
            expand.remove(self.goal)

        count = 0
        if isinstance(self.env, Grid3D):
            for x in expand:
                count += 1
                self.ax.scatter(x[0], x[1], x[2], s=100, color='blue', marker='s', alpha=0.2)
                plt.gcf().canvas.mpl_connect('key_release_event',
                                            lambda event: [exit(0) if event.key == 'escape' else None])
                if count < len(expand) / 3:         length = 20
                elif count < len(expand) * 2 / 3:   length = 30
                else:                               length = 40
                if count % length == 0:             plt.pause(0.00001)
        
        plt.pause(0.0001)

    def plotPath(self, path: list, path_color: str='#13ae00', path_style: str="-") -> None:
        '''
        Plot path in global planning.

        Parameters
        ----------
        path: Path found in global planning
        '''
        # Extract path coordinates
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]
        path_z = [path[i][2] for i in range(len(path))]

        # Make them denser
        # for plot, we want a dense map so devide the z-axis by 10
        # new_path_z = []
        # for z in path_z:
        #     new_path_z.append(z / 10.0)

        # Plot path
        plt.plot(path_x, path_y, path_z, path_style, linewidth=2, color=path_color)


    def plotAgent(self, pose: tuple, radius: float=1) -> None:
        '''
        Plot agent with specifical pose.

        Parameters
        ----------
        pose: Pose of agent
        radius: Radius of agent
        '''
        x, y, theta = pose
        ref_vec = np.array([[radius / 2], [0]])
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
        end_pt = rot_mat @ ref_vec + np.array([[x], [y]])

        try:
            self.ax.artists.pop()
            for art in self.ax.get_children():
                if isinstance(art, matplotlib.patches.FancyArrow):
                    art.remove()
        except:
            pass

        self.ax.arrow(x, y, float(end_pt[0]) - x, float(end_pt[1]) - y,
                width=0.1, head_width=0.40, color="r")
        circle = plt.Circle((x, y), radius, color="r", fill=False)
        self.ax.add_artist(circle)

    def plotHistoryPose(self, history_pose, predict_path=None, lookahead_pts=None) -> None:
        lookahead_handler = None
        for i, pose in enumerate(history_pose):
            if i < len(history_pose) - 1:
                plt.plot([history_pose[i][0], history_pose[i + 1][0]],
                    [history_pose[i][1], history_pose[i + 1][1]], c="#13ae00")
                if predict_path is not None:
                    plt.plot(predict_path[i][:, 0], predict_path[i][:, 1], c="#ddd")
            i += 1

            # agent
            self.plotAgent(pose)

            # lookahead
            if lookahead_handler is not None:
                lookahead_handler.remove()
            if lookahead_pts is not None:
                try:
                    lookahead_handler = self.ax.scatter(lookahead_pts[i][0], lookahead_pts[i][1], c="b")
                except:
                    lookahead_handler = self.ax.scatter(lookahead_pts[-1][0], lookahead_pts[-1][1], c="b")

            plt.gcf().canvas.mpl_connect('key_release_event',
                                        lambda event: [exit(0) if event.key == 'escape' else None])
            if i % 5 == 0:             plt.pause(0.03)

    def plotCostCurve(self, cost_list: list, name: str) -> None:
        '''
        Plot cost curve with epochs using in evolutionary searching.

        Parameters
        ----------
        cost_list: Cost with epochs
        name: Algorithm name or some other information
        '''
        plt.plot(cost_list, color="b")
        plt.xlabel("epochs")
        plt.ylabel("cost value")
        plt.title(name)
        plt.grid()

    def plotEllipse(self, ellipse: np.ndarray, color: str = 'darkorange', linestyle: str = '--', linewidth: float = 2):
        plt.plot(ellipse[0, :], ellipse[1, :], linestyle=linestyle, color=color, linewidth=linewidth)

    def connect(self, name: str, func) -> None:
        self.fig.canvas.mpl_connect(name, func)

    def clean(self):
        plt.cla()

    def update(self):
        self.fig.canvas.draw_idle()

    @staticmethod
    def plotArrow(x, y, theta, length, color):
        angle = np.deg2rad(30)
        d = 0.5 * length
        w = 2

        x_start, y_start = x, y
        x_end = x + length * np.cos(theta)
        y_end = y + length * np.sin(theta)

        theta_hat_L = theta + np.pi - angle
        theta_hat_R = theta + np.pi + angle

        x_hat_start = x_end
        x_hat_end_L = x_hat_start + d * np.cos(theta_hat_L)
        x_hat_end_R = x_hat_start + d * np.cos(theta_hat_R)

        y_hat_start = y_end
        y_hat_end_L = y_hat_start + d * np.sin(theta_hat_L)
        y_hat_end_R = y_hat_start + d * np.sin(theta_hat_R)

        plt.plot([x_start, x_end], [y_start, y_end], color=color, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_L], [y_hat_start, y_hat_end_L], color=color, linewidth=w)
        plt.plot([x_hat_start, x_hat_end_R], [y_hat_start, y_hat_end_R], color=color, linewidth=w)

    @staticmethod
    def plotCar(x, y, theta, width, length, color):
        theta_B = np.pi + theta

        xB = x + length / 4 * np.cos(theta_B)
        yB = y + length / 4 * np.sin(theta_B)

        theta_BL = theta_B + np.pi / 2
        theta_BR = theta_B - np.pi / 2

        x_BL = xB + width / 2 * np.cos(theta_BL)        # Bottom-Left vertex
        y_BL = yB + width / 2 * np.sin(theta_BL)
        x_BR = xB + width / 2 * np.cos(theta_BR)        # Bottom-Right vertex
        y_BR = yB + width / 2 * np.sin(theta_BR)

        x_FL = x_BL + length * np.cos(theta)               # Front-Left vertex
        y_FL = y_BL + length * np.sin(theta)
        x_FR = x_BR + length * np.cos(theta)               # Front-Right vertex
        y_FR = y_BR + length * np.sin(theta)

        plt.plot([x_BL, x_BR, x_FR, x_FL, x_BL],
                 [y_BL, y_BR, y_FR, y_FL, y_BL],
                 linewidth=1, color=color)

        Plot3D.plotArrow(x, y, theta, length / 2, color)