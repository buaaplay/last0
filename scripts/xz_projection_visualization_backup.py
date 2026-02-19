import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class XZProjectionVisualizer:
    def __init__(self):
        self.trajectories_0 = []
        self.trajectories_400 = []
        self.absolute_trajectories_0 = []
        self.absolute_trajectories_400 = []
    
    def load_json_data(self, filename):
        """加载JSON数据"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def extract_successful_trajectories(self, data):
        actions = data['actions']
        successes = data.get('successes', [])
        trajectories = []
        successful_indices = []
        
        for i, action_sequence in enumerate(actions):
            if i < len(successes) and successes[i]:
                trajectory = []
                for point in action_sequence:
                    if len(point) >= 3:
                        trajectory.append([point[0], point[1], point[2]])
                if trajectory:
                    trajectories.append(np.array(trajectory))
                    successful_indices.append(i)
        
        return trajectories, successful_indices
    
    def convert_relative_to_absolute(self, relative_trajectories):
        """将相对位置坐标转换为绝对位置"""
        absolute_trajectories = []
        
        for trajectory in relative_trajectories:
            absolute_trajectory = np.zeros_like(trajectory)
            absolute_trajectory[0] = trajectory[0]  # 第一个点是起始点
            
            # 累积计算绝对位置
            for i in range(1, len(trajectory)):
                absolute_trajectory[i] = absolute_trajectory[i-1] + trajectory[i]
            
            absolute_trajectories.append(absolute_trajectory)
        
        return absolute_trajectories
    
    def calculate_unified_axis_limits(self, trajectories_0, trajectories_400):
        """计算统一的轴限制"""
        all_trajectories = trajectories_0 + trajectories_400
        all_x_coords = np.concatenate([traj[:, 0] for traj in all_trajectories])
        all_y_coords = np.concatenate([traj[:, 1] for traj in all_trajectories])
        all_z_coords = np.concatenate([traj[:, 2] for traj in all_trajectories])
        
        x_min, x_max = all_x_coords.min(), all_x_coords.max()
        y_min, y_max = all_y_coords.min(), all_y_coords.max()
        z_min, z_max = all_z_coords.min(), all_z_coords.max()
        
        # 添加一些边距
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        z_margin = (z_max - z_min) * 0.05
        
        x_lim = [x_min - x_margin, x_max + x_margin]
        y_lim = [y_min - y_margin, y_max + y_margin]
        # z_lim = [z_min - z_margin, z_max + z_margin]
        z_lim = [-30, z_max + z_margin]

        
        return x_lim, y_lim, z_lim
    
    def visualize_xz_yz_projection_points(self, trajectories_0, trajectories_400, 
                                         x_lim, z_lim, y_lim):
        """同时可视化X-Z和Y-Z投影点图"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 合并所有点数据
        all_points_0_xz = np.vstack([traj[:, [0, 2]] for traj in trajectories_0])
        all_points_400_xz = np.vstack([traj[:, [0, 2]] for traj in trajectories_400])
        all_points_0_yz = np.vstack([traj[:, [1, 2]] for traj in trajectories_0])
        all_points_400_yz = np.vstack([traj[:, [1, 2]] for traj in trajectories_400])
        
        # 第一行：X-Z投影
        # 1. RLVR_0 的 X-Z 投影点图
        ax1 = axes[0, 0]
        ax1.scatter(all_points_0_xz[:, 0], all_points_0_xz[:, 1], 
                   color='#1f77b4', s=15, alpha=0.6)
        ax1.set_title('RLVR_0 - X-Z投影 (点图)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X坐标', fontsize=12)
        ax1.set_ylabel('Z坐标', fontsize=12)
        ax1.set_xlim(x_lim)
        ax1.set_ylim(z_lim)
        ax1.grid(True, alpha=0.3)
        
        # 2. RLVR_400 的 X-Z 投影点图
        ax2 = axes[0, 1]
        ax2.scatter(all_points_400_xz[:, 0], all_points_400_xz[:, 1], 
                   color='#ff6b6b', s=15, alpha=0.6)
        ax2.set_title('RLVR_400 - X-Z投影 (点图)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X坐标', fontsize=12)
        ax2.set_ylabel('Z坐标', fontsize=12)
        ax2.set_xlim(x_lim)
        ax2.set_ylim(z_lim)
        ax2.grid(True, alpha=0.3)
        
        # 3. X-Z投影叠加对比图
        ax3 = axes[0, 2]
        ax3.scatter(all_points_0_xz[:, 0], all_points_0_xz[:, 1], 
                   color='#1f77b4', s=10, alpha=0.5, label='RLVR_0')
        ax3.scatter(all_points_400_xz[:, 0], all_points_400_xz[:, 1], 
                   color='#ff6b6b', s=10, alpha=0.5, label='RLVR_400')
        ax3.set_title('RLVR_0 vs RLVR_400 - X-Z投影对比', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X坐标', fontsize=12)
        ax3.set_ylabel('Z坐标', fontsize=12)
        ax3.set_xlim(x_lim)
        ax3.set_ylim(z_lim)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 第二行：Y-Z投影
        # 4. RLVR_0 的 Y-Z 投影点图
        ax4 = axes[1, 0]
        ax4.scatter(all_points_0_yz[:, 0], all_points_0_yz[:, 1], 
                   color='#1f77b4', s=15, alpha=0.6)
        ax4.set_title('RLVR_0 - Y-Z投影 (点图)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Y坐标', fontsize=12)
        ax4.set_ylabel('Z坐标', fontsize=12)
        ax4.set_xlim(y_lim)
        ax4.set_ylim(z_lim)
        ax4.grid(True, alpha=0.3)
        
        # 5. RLVR_400 的 Y-Z 投影点图
        ax5 = axes[1, 1]
        ax5.scatter(all_points_400_yz[:, 0], all_points_400_yz[:, 1], 
                   color='#ff6b6b', s=15, alpha=0.6)
        ax5.set_title('RLVR_400 - Y-Z投影 (点图)', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Y坐标', fontsize=12)
        ax5.set_ylabel('Z坐标', fontsize=12)
        ax5.set_xlim(y_lim)
        ax5.set_ylim(z_lim)
        ax5.grid(True, alpha=0.3)
        
        # 6. Y-Z投影叠加对比图
        ax6 = axes[1, 2]
        ax6.scatter(all_points_0_yz[:, 0], all_points_0_yz[:, 1], 
                   color='#1f77b4', s=10, alpha=0.5, label='RLVR_0')
        ax6.scatter(all_points_400_yz[:, 0], all_points_400_yz[:, 1], 
                   color='#ff6b6b', s=10, alpha=0.5, label='RLVR_400')
        ax6.set_title('RLVR_0 vs RLVR_400 - Y-Z投影对比', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Y坐标', fontsize=12)
        ax6.set_ylabel('Z坐标', fontsize=12)
        ax6.set_xlim(y_lim)
        ax6.set_ylim(z_lim)
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        plt.tight_layout()
        plt.show()
    
    def create_hexbin_density_plot(self, trajectories_0, trajectories_400, 
                                  x_lim, z_lim):
        """创建六边形密度图"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 合并所有点数据
        all_points_0 = np.vstack([traj[:, [0, 2]] for traj in trajectories_0])
        all_points_400 = np.vstack([traj[:, [0, 2]] for traj in trajectories_400])
        
        # 1. RLVR_0 六边形密度图
        ax1 = axes[0]
        hb1 = ax1.hexbin(all_points_0[:, 0], all_points_0[:, 1], 
                        gridsize=30, cmap='Blues', alpha=0.8)
        ax1.set_title('RLVR_0 - X-Z六边形密度分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X坐标', fontsize=12)
        ax1.set_ylabel('Z坐标', fontsize=12)
        ax1.set_xlim(x_lim)
        ax1.set_ylim(z_lim)
        ax1.grid(True, alpha=0.3)
        plt.colorbar(hb1, ax=ax1, label='点密度')
        
        # 2. RLVR_400 六边形密度图
        ax2 = axes[1]
        hb2 = ax2.hexbin(all_points_400[:, 0], all_points_400[:, 1], 
                        gridsize=30, cmap='Reds', alpha=0.8)
        ax2.set_title('RLVR_400 - X-Z六边形密度分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X坐标', fontsize=12)
        ax2.set_ylabel('Z坐标', fontsize=12)
        ax2.set_xlim(x_lim)
        ax2.set_ylim(z_lim)
        ax2.grid(True, alpha=0.3)
        plt.colorbar(hb2, ax=ax2, label='点密度')
        
        # 3. 叠加六边形密度图
        ax3 = axes[2]
        hb3_0 = ax3.hexbin(all_points_0[:, 0], all_points_0[:, 1], 
                          gridsize=30, cmap='Blues', alpha=0.6)
        hb3_400 = ax3.hexbin(all_points_400[:, 0], all_points_400[:, 1], 
                            gridsize=30, cmap='Reds', alpha=0.6)
        ax3.set_title('RLVR_0 vs RLVR_400 - 六边形密度对比', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X坐标', fontsize=12)
        ax3.set_ylabel('Z坐标', fontsize=12)
        ax3.set_xlim(x_lim)
        ax3.set_ylim(z_lim)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_kde_density_plot(self, trajectories_0, trajectories_400, 
                               x_lim, z_lim):
        """创建KDE散点密度图"""
        from scipy.stats import gaussian_kde
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 合并所有点数据
        all_points_0 = np.vstack([traj[:, [0, 2]] for traj in trajectories_0])
        all_points_400 = np.vstack([traj[:, [0, 2]] for traj in trajectories_400])
        
        # 创建网格用于密度估计
        x_range = np.linspace(x_lim[0], x_lim[1], 100)
        z_range = np.linspace(z_lim[0], z_lim[1], 100)
        X, Z = np.meshgrid(x_range, z_range)
        grid_points = np.vstack([X.ravel(), Z.ravel()])
        
        # 1. RLVR_0 KDE密度图
        ax1 = axes[0]
        if len(all_points_0) > 0:
            kde_0 = gaussian_kde(all_points_0.T)
            density_0 = kde_0(grid_points).reshape(X.shape)
            im1 = ax1.contourf(X, Z, density_0, levels=20, cmap='Blues', alpha=0.8)
            ax1.scatter(all_points_0[:, 0], all_points_0[:, 1], 
                       color='darkblue', s=1, alpha=0.3)
        ax1.set_title('RLVR_0 - X-Z KDE密度分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X坐标', fontsize=12)
        ax1.set_ylabel('Z坐标', fontsize=12)
        ax1.set_xlim(x_lim)
        ax1.set_ylim(z_lim)
        ax1.grid(True, alpha=0.3)
        if len(all_points_0) > 0:
            plt.colorbar(im1, ax=ax1, label='密度')
        
        # 2. RLVR_400 KDE密度图
        ax2 = axes[1]
        if len(all_points_400) > 0:
            kde_400 = gaussian_kde(all_points_400.T)
            density_400 = kde_400(grid_points).reshape(X.shape)
            im2 = ax2.contourf(X, Z, density_400, levels=20, cmap='Reds', alpha=0.8)
            ax2.scatter(all_points_400[:, 0], all_points_400[:, 1], 
                       color='darkred', s=1, alpha=0.3)
        ax2.set_title('RLVR_400 - X-Z KDE密度分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X坐标', fontsize=12)
        ax2.set_ylabel('Z坐标', fontsize=12)
        ax2.set_xlim(x_lim)
        ax2.set_ylim(z_lim)
        ax2.grid(True, alpha=0.3)
        if len(all_points_400) > 0:
            plt.colorbar(im2, ax=ax2, label='密度')
        
        # 3. 叠加KDE密度图
        ax3 = axes[2]
        if len(all_points_0) > 0 and len(all_points_400) > 0:
            # 叠加两个密度图
            density_combined = density_0 + density_400
            im3 = ax3.contourf(X, Z, density_combined, levels=20, cmap='viridis', alpha=0.8)
            ax3.scatter(all_points_0[:, 0], all_points_0[:, 1], 
                       color='blue', s=1, alpha=0.3, label='RLVR_0')
            ax3.scatter(all_points_400[:, 0], all_points_400[:, 1], 
                       color='red', s=1, alpha=0.3, label='RLVR_400')
            ax3.legend()
            plt.colorbar(im3, ax=ax3, label='组合密度')
        ax3.set_title('RLVR_0 vs RLVR_400 - KDE密度对比', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X坐标', fontsize=12)
        ax3.set_ylabel('Z坐标', fontsize=12)
        ax3.set_xlim(x_lim)
        ax3.set_ylim(z_lim)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_contour_plot(self, trajectories_0, trajectories_400, 
                           x_lim, z_lim):
        """创建等高线图"""
        from scipy.stats import gaussian_kde
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 合并所有点数据
        all_points_0 = np.vstack([traj[:, [0, 2]] for traj in trajectories_0])
        all_points_400 = np.vstack([traj[:, [0, 2]] for traj in trajectories_400])
        
        # 创建网格用于密度估计
        x_range = np.linspace(x_lim[0], x_lim[1], 100)
        z_range = np.linspace(z_lim[0], z_lim[1], 100)
        X, Z = np.meshgrid(x_range, z_range)
        grid_points = np.vstack([X.ravel(), Z.ravel()])
        
        # 1. RLVR_0 等高线图
        ax1 = axes[0]
        if len(all_points_0) > 0:
            kde_0 = gaussian_kde(all_points_0.T)
            density_0 = kde_0(grid_points).reshape(X.shape)
            contours1 = ax1.contour(X, Z, density_0, levels=10, colors='blue', alpha=0.8, linewidths=2)
            ax1.clabel(contours1, inline=True, fontsize=8)
            ax1.scatter(all_points_0[:, 0], all_points_0[:, 1], 
                       color='lightblue', s=5, alpha=0.5)
        ax1.set_title('RLVR_0 - X-Z等高线分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X坐标', fontsize=12)
        ax1.set_ylabel('Z坐标', fontsize=12)
        ax1.set_xlim(x_lim)
        ax1.set_ylim(z_lim)
        ax1.grid(True, alpha=0.3)
        
        # 2. RLVR_400 等高线图
        ax2 = axes[1]
        if len(all_points_400) > 0:
            kde_400 = gaussian_kde(all_points_400.T)
            density_400 = kde_400(grid_points).reshape(X.shape)
            contours2 = ax2.contour(X, Z, density_400, levels=10, colors='red', alpha=0.8, linewidths=2)
            ax2.clabel(contours2, inline=True, fontsize=8)
            ax2.scatter(all_points_400[:, 0], all_points_400[:, 1], 
                       color='lightcoral', s=5, alpha=0.5)
        ax2.set_title('RLVR_400 - X-Z等高线分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X坐标', fontsize=12)
        ax2.set_ylabel('Z坐标', fontsize=12)
        ax2.set_xlim(x_lim)
        ax2.set_ylim(z_lim)
        ax2.grid(True, alpha=0.3)
        
        # 3. 叠加等高线图
        ax3 = axes[2]
        if len(all_points_0) > 0 and len(all_points_400) > 0:
            contours3_0 = ax3.contour(X, Z, density_0, levels=8, colors='blue', alpha=0.6, linewidths=1.5, linestyles='-')
            contours3_400 = ax3.contour(X, Z, density_400, levels=8, colors='red', alpha=0.6, linewidths=1.5, linestyles='--')
            ax3.scatter(all_points_0[:, 0], all_points_0[:, 1], 
                       color='lightblue', s=3, alpha=0.4, label='RLVR_0')
            ax3.scatter(all_points_400[:, 0], all_points_400[:, 1], 
                       color='lightcoral', s=3, alpha=0.4, label='RLVR_400')
            ax3.legend()
        ax3.set_title('RLVR_0 vs RLVR_400 - 等高线对比', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X坐标', fontsize=12)
        ax3.set_ylabel('Z坐标', fontsize=12)
        ax3.set_xlim(x_lim)
        ax3.set_ylim(z_lim)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_convex_hull_plot(self, trajectories_0, trajectories_400, 
                               x_lim, z_lim):
        """创建凸包可视化图"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 合并所有点数据
        all_points_0 = np.vstack([traj[:, [0, 2]] for traj in trajectories_0])
        all_points_400 = np.vstack([traj[:, [0, 2]] for traj in trajectories_400])
        
        # 1. RLVR_0 凸包图
        ax1 = axes[0]
        if len(all_points_0) > 3:  # 需要至少4个点才能形成凸包
            hull_0 = ConvexHull(all_points_0)
            hull_points_0 = all_points_0[hull_0.vertices]
            hull_polygon_0 = Polygon(hull_points_0)
            
            # 绘制凸包
            hull_x, hull_y = hull_polygon_0.exterior.xy
            ax1.plot(hull_x, hull_y, 'b-', linewidth=3, alpha=0.8, label='凸包边界')
            ax1.fill(hull_x, hull_y, color='blue', alpha=0.2, label='凸包区域')
            
            # 绘制所有点
            ax1.scatter(all_points_0[:, 0], all_points_0[:, 1], 
                       color='darkblue', s=8, alpha=0.6)
            
            # 绘制凸包顶点
            ax1.scatter(hull_points_0[:, 0], hull_points_0[:, 1], 
                       color='red', s=50, marker='s', edgecolors='black', linewidth=1)
            
            area_0 = hull_polygon_0.area
            ax1.set_title(f'RLVR_0 - 凸包分布\n面积: {area_0:.3f}', fontsize=14, fontweight='bold')
        else:
            ax1.scatter(all_points_0[:, 0], all_points_0[:, 1], 
                       color='darkblue', s=8, alpha=0.6)
            ax1.set_title('RLVR_0 - 点数不足形成凸包', fontsize=14, fontweight='bold')
        
        ax1.set_xlabel('X坐标', fontsize=12)
        ax1.set_ylabel('Z坐标', fontsize=12)
        ax1.set_xlim(x_lim)
        ax1.set_ylim(z_lim)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. RLVR_400 凸包图
        ax2 = axes[1]
        if len(all_points_400) > 3:
            hull_400 = ConvexHull(all_points_400)
            hull_points_400 = all_points_400[hull_400.vertices]
            hull_polygon_400 = Polygon(hull_points_400)
            
            # 绘制凸包
            hull_x, hull_y = hull_polygon_400.exterior.xy
            ax2.plot(hull_x, hull_y, 'r-', linewidth=3, alpha=0.8, label='凸包边界')
            ax2.fill(hull_x, hull_y, color='red', alpha=0.2, label='凸包区域')
            
            # 绘制所有点
            ax2.scatter(all_points_400[:, 0], all_points_400[:, 1], 
                       color='darkred', s=8, alpha=0.6)
            
            # 绘制凸包顶点
            ax2.scatter(hull_points_400[:, 0], hull_points_400[:, 1], 
                       color='blue', s=50, marker='s', edgecolors='black', linewidth=1)
            
            area_400 = hull_polygon_400.area
            ax2.set_title(f'RLVR_400 - 凸包分布\n面积: {area_400:.3f}', fontsize=14, fontweight='bold')
        else:
            ax2.scatter(all_points_400[:, 0], all_points_400[:, 1], 
                       color='darkred', s=8, alpha=0.6)
            ax2.set_title('RLVR_400 - 点数不足形成凸包', fontsize=14, fontweight='bold')
        
        ax2.set_xlabel('X坐标', fontsize=12)
        ax2.set_ylabel('Z坐标', fontsize=12)
        ax2.set_xlim(x_lim)
        ax2.set_ylim(z_lim)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 叠加凸包对比图
        ax3 = axes[2]
        
        # 绘制RLVR_0的凸包
        if len(all_points_0) > 3:
            hull_0 = ConvexHull(all_points_0)
            hull_points_0 = all_points_0[hull_0.vertices]
            hull_polygon_0 = Polygon(hull_points_0)
            
            hull_x, hull_y = hull_polygon_0.exterior.xy
            ax3.plot(hull_x, hull_y, 'b-', linewidth=2, alpha=0.8, label='RLVR_0 凸包')
            ax3.fill(hull_x, hull_y, color='blue', alpha=0.15)
            ax3.scatter(all_points_0[:, 0], all_points_0[:, 1], 
                       color='lightblue', s=5, alpha=0.4)
        
        # 绘制RLVR_400的凸包
        if len(all_points_400) > 3:
            hull_400 = ConvexHull(all_points_400)
            hull_points_400 = all_points_400[hull_400.vertices]
            hull_polygon_400 = Polygon(hull_points_400)
            
            hull_x, hull_y = hull_polygon_400.exterior.xy
            ax3.plot(hull_x, hull_y, 'r-', linewidth=2, alpha=0.8, label='RLVR_400 凸包')
            ax3.fill(hull_x, hull_y, color='red', alpha=0.15)
            ax3.scatter(all_points_400[:, 0], all_points_400[:, 1], 
                       color='lightcoral', s=5, alpha=0.4)
        
        ax3.set_title('RLVR_0 vs RLVR_400 - 凸包对比', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X坐标', fontsize=12)
        ax3.set_ylabel('Z坐标', fontsize=12)
        ax3.set_xlim(x_lim)
        ax3.set_ylim(z_lim)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 返回凸包面积用于统计
        areas = {}
        if len(all_points_0) > 3:
            areas['RLVR_0'] = hull_polygon_0.area
        if len(all_points_400) > 3:
            areas['RLVR_400'] = hull_polygon_400.area
        
        return areas
    
    def create_alpha_shape_plot(self, trajectories_0, trajectories_400, 
                               x_lim, z_lim, alpha_value=None):
        """创建Alpha Shape可视化图"""
        try:
            from alphashape import alphashape
        except ImportError:
            print("警告: alphashape库未安装，将使用简化版本")
            return self._create_simplified_alpha_shape_plot(trajectories_0, trajectories_400, x_lim, z_lim)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 合并所有点数据
        all_points_0 = np.vstack([traj[:, [0, 2]] for traj in trajectories_0])
        all_points_400 = np.vstack([traj[:, [0, 2]] for traj in trajectories_400])
        
        areas = {}
        
        # 1. RLVR_0 Alpha Shape图
        ax1 = axes[0]
        if len(all_points_0) > 2:
            try:
                alpha_shape_0 = alphashape(all_points_0, alpha_value)
                if alpha_shape_0 is not None:
                    # 绘制Alpha Shape
                    if hasattr(alpha_shape_0, 'exterior'):
                        x, y = alpha_shape_0.exterior.xy
                        ax1.plot(x, y, 'b-', linewidth=3, alpha=0.8, label='Alpha Shape边界')
                        ax1.fill(x, y, color='blue', alpha=0.2, label='Alpha Shape区域')
                        areas['RLVR_0'] = alpha_shape_0.area
                    else:
                        # 处理多点集合
                        for geom in alpha_shape_0.geoms:
                            if hasattr(geom, 'exterior'):
                                x, y = geom.exterior.xy
                                ax1.plot(x, y, 'b-', linewidth=3, alpha=0.8)
                                ax1.fill(x, y, color='blue', alpha=0.2)
                                areas['RLVR_0'] = areas.get('RLVR_0', 0) + geom.area
                else:
                    print("RLVR_0: 无法生成Alpha Shape")
                
                # 绘制所有点
                ax1.scatter(all_points_0[:, 0], all_points_0[:, 1], 
                           color='darkblue', s=8, alpha=0.6)
                
                area_text = f"面积: {areas.get('RLVR_0', 0):.3f}" if 'RLVR_0' in areas else "无法计算面积"
                ax1.set_title(f'RLVR_0 - Alpha Shape分布\n{area_text}', fontsize=14, fontweight='bold')
            except Exception as e:
                print(f"RLVR_0 Alpha Shape错误: {e}")
                ax1.scatter(all_points_0[:, 0], all_points_0[:, 1], 
                           color='darkblue', s=8, alpha=0.6)
                ax1.set_title('RLVR_0 - Alpha Shape计算失败', fontsize=14, fontweight='bold')
        else:
            ax1.scatter(all_points_0[:, 0], all_points_0[:, 1], 
                       color='darkblue', s=8, alpha=0.6)
            ax1.set_title('RLVR_0 - 点数不足', fontsize=14, fontweight='bold')
        
        ax1.set_xlabel('X坐标', fontsize=12)
        ax1.set_ylabel('Z坐标', fontsize=12)
        ax1.set_xlim(x_lim)
        ax1.set_ylim(z_lim)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. RLVR_400 Alpha Shape图
        ax2 = axes[1]
        if len(all_points_400) > 2:
            try:
                alpha_shape_400 = alphashape(all_points_400, alpha_value)
                if alpha_shape_400 is not None:
                    # 绘制Alpha Shape
                    if hasattr(alpha_shape_400, 'exterior'):
                        x, y = alpha_shape_400.exterior.xy
                        ax2.plot(x, y, 'r-', linewidth=3, alpha=0.8, label='Alpha Shape边界')
                        ax2.fill(x, y, color='red', alpha=0.2, label='Alpha Shape区域')
                        areas['RLVR_400'] = alpha_shape_400.area
                    else:
                        # 处理多点集合
                        for geom in alpha_shape_400.geoms:
                            if hasattr(geom, 'exterior'):
                                x, y = geom.exterior.xy
                                ax2.plot(x, y, 'r-', linewidth=3, alpha=0.8)
                                ax2.fill(x, y, color='red', alpha=0.2)
                                areas['RLVR_400'] = areas.get('RLVR_400', 0) + geom.area
                else:
                    print("RLVR_400: 无法生成Alpha Shape")
                
                # 绘制所有点
                ax2.scatter(all_points_400[:, 0], all_points_400[:, 1], 
                           color='darkred', s=8, alpha=0.6)
                
                area_text = f"面积: {areas.get('RLVR_400', 0):.3f}" if 'RLVR_400' in areas else "无法计算面积"
                ax2.set_title(f'RLVR_400 - Alpha Shape分布\n{area_text}', fontsize=14, fontweight='bold')
            except Exception as e:
                print(f"RLVR_400 Alpha Shape错误: {e}")
                ax2.scatter(all_points_400[:, 0], all_points_400[:, 1], 
                           color='darkred', s=8, alpha=0.6)
                ax2.set_title('RLVR_400 - Alpha Shape计算失败', fontsize=14, fontweight='bold')
        else:
            ax2.scatter(all_points_400[:, 0], all_points_400[:, 1], 
                       color='darkred', s=8, alpha=0.6)
            ax2.set_title('RLVR_400 - 点数不足', fontsize=14, fontweight='bold')
        
        ax2.set_xlabel('X坐标', fontsize=12)
        ax2.set_ylabel('Z坐标', fontsize=12)
        ax2.set_xlim(x_lim)
        ax2.set_ylim(z_lim)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 叠加Alpha Shape对比图
        ax3 = axes[2]
        
        # 绘制RLVR_0的Alpha Shape
        if len(all_points_0) > 2:
            try:
                alpha_shape_0 = alphashape(all_points_0, alpha_value)
                if alpha_shape_0 is not None:
                    if hasattr(alpha_shape_0, 'exterior'):
                        x, y = alpha_shape_0.exterior.xy
                        ax3.plot(x, y, 'b-', linewidth=2, alpha=0.8, label='RLVR_0 Alpha Shape')
                        ax3.fill(x, y, color='blue', alpha=0.15)
                    else:
                        for geom in alpha_shape_0.geoms:
                            if hasattr(geom, 'exterior'):
                                x, y = geom.exterior.xy
                                ax3.plot(x, y, 'b-', linewidth=2, alpha=0.8)
                                ax3.fill(x, y, color='blue', alpha=0.15)
                    ax3.scatter(all_points_0[:, 0], all_points_0[:, 1], 
                               color='lightblue', s=5, alpha=0.4)
            except:
                ax3.scatter(all_points_0[:, 0], all_points_0[:, 1], 
                           color='lightblue', s=5, alpha=0.4)
        
        # 绘制RLVR_400的Alpha Shape
        if len(all_points_400) > 2:
            try:
                alpha_shape_400 = alphashape(all_points_400, alpha_value)
                if alpha_shape_400 is not None:
                    if hasattr(alpha_shape_400, 'exterior'):
                        x, y = alpha_shape_400.exterior.xy
                        ax3.plot(x, y, 'r-', linewidth=2, alpha=0.8, label='RLVR_400 Alpha Shape')
                        ax3.fill(x, y, color='red', alpha=0.15)
                    else:
                        for geom in alpha_shape_400.geoms:
                            if hasattr(geom, 'exterior'):
                                x, y = geom.exterior.xy
                                ax3.plot(x, y, 'r-', linewidth=2, alpha=0.8)
                                ax3.fill(x, y, color='red', alpha=0.15)
                    ax3.scatter(all_points_400[:, 0], all_points_400[:, 1], 
                               color='lightcoral', s=5, alpha=0.4)
            except:
                ax3.scatter(all_points_400[:, 0], all_points_400[:, 1], 
                           color='lightcoral', s=5, alpha=0.4)
        
        ax3.set_title('RLVR_0 vs RLVR_400 - Alpha Shape对比', fontsize=14, fontweight='bold')
        ax3.set_xlabel('X坐标', fontsize=12)
        ax3.set_ylabel('Z坐标', fontsize=12)
        ax3.set_xlim(x_lim)
        ax3.set_ylim(z_lim)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()
        
        return areas
    
    def _create_simplified_alpha_shape_plot(self, trajectories_0, trajectories_400, x_lim, z_lim):
        """简化版Alpha Shape图（当alphashape库不可用时）"""
        print("使用简化版Alpha Shape可视化")
        return self.create_convex_hull_plot(trajectories_0, trajectories_400, x_lim, z_lim)
    
    def print_statistics(self, trajectories_0, trajectories_400, x_lim, z_lim):
        """打印统计信息"""
        print("=" * 60)
        print("X-Z投影统计信息")
        print("=" * 60)
        
        print(f"\nRLVR_0:")
        print(f"  轨迹数量: {len(trajectories_0)}")
        print(f"  总点数: {sum(len(traj) for traj in trajectories_0)}")
        
        all_points_0 = np.vstack([traj[:, [0, 2]] for traj in trajectories_0])
        print(f"  X范围: [{all_points_0[:, 0].min():.3f}, {all_points_0[:, 0].max():.3f}]")
        print(f"  Z范围: [{all_points_0[:, 1].min():.3f}, {all_points_0[:, 1].max():.3f}]")
        
        print(f"\nRLVR_400:")
        print(f"  轨迹数量: {len(trajectories_400)}")
        print(f"  总点数: {sum(len(traj) for traj in trajectories_400)}")
        
        all_points_400 = np.vstack([traj[:, [0, 2]] for traj in trajectories_400])
        print(f"  X范围: [{all_points_400[:, 0].min():.3f}, {all_points_400[:, 0].max():.3f}]")
        print(f"  Z范围: [{all_points_400[:, 1].min():.3f}, {all_points_400[:, 1].max():.3f}]")
        
        print(f"\n统一轴范围:")
        print(f"  X轴: [{x_lim[0]:.3f}, {x_lim[1]:.3f}]")
        print(f"  Z轴: [{z_lim[0]:.3f}, {z_lim[1]:.3f}]")
        
        print("=" * 60)

def main():
    # 初始化可视化器
    visualizer = XZProjectionVisualizer()
    
    # 文件列表
    json_files = {
        'task_4_sft150000_rlvr_0.json': 'RLVR_0',
        'task_4_sft150000_rlvr_400.json': 'RLVR_400'
    }
    
    print("正在加载和处理数据...")
    
    # 处理每个文件
    for filename, name in json_files.items():
        print(f"\n处理文件: {filename}")
        
        # 加载数据
        data = visualizer.load_json_data(filename)
        
        # 提取成功轨迹
        trajectories, successful_indices = visualizer.extract_successful_trajectories(data)
        print(f"  提取到 {len(trajectories)} 条成功轨迹")
        
        # 转换为绝对位置
        absolute_trajectories = visualizer.convert_relative_to_absolute(trajectories)
        print(f"  转换为绝对位置完成")
        
        # 存储数据
        if name == 'RLVR_0':
            visualizer.trajectories_0 = trajectories
            visualizer.absolute_trajectories_0 = absolute_trajectories
        else:
            visualizer.trajectories_400 = trajectories
            visualizer.absolute_trajectories_400 = absolute_trajectories
    
    # 计算统一的轴限制
    x_lim, y_lim, z_lim = visualizer.calculate_unified_axis_limits(
        visualizer.absolute_trajectories_0, 
        visualizer.absolute_trajectories_400
    )
    
    # 打印统计信息
    visualizer.print_statistics(
        visualizer.absolute_trajectories_0, 
        visualizer.absolute_trajectories_400, 
        x_lim, z_lim
    )
    
    # 创建X-Z和Y-Z投影点图可视化
    print("\n创建X-Z和Y-Z投影点图可视化...")
    visualizer.visualize_xz_yz_projection_points(
        visualizer.absolute_trajectories_0, 
        visualizer.absolute_trajectories_400, 
        x_lim, z_lim, y_lim
    )
    
    # 创建六边形密度图可视化
    print("\n创建六边形密度分布图...")
    visualizer.create_hexbin_density_plot(
        visualizer.absolute_trajectories_0, 
        visualizer.absolute_trajectories_400, 
        x_lim, z_lim
    )
    
    # 创建KDE散点密度图可视化
    print("\n创建KDE散点密度分布图...")
    visualizer.create_kde_density_plot(
        visualizer.absolute_trajectories_0, 
        visualizer.absolute_trajectories_400, 
        x_lim, z_lim
    )
    
    # 创建等高线图可视化
    print("\n创建等高线分布图...")
    visualizer.create_contour_plot(
        visualizer.absolute_trajectories_0, 
        visualizer.absolute_trajectories_400, 
        x_lim, z_lim
    )
    

    
    # 打印覆盖区域统计信息
    print("\n" + "=" * 60)
    print("覆盖区域统计信息")
    print("=" * 60)
    
    
    print("=" * 60)
    
    print("\nX-Z投影可视化完成！")

if __name__ == "__main__":
    main()
