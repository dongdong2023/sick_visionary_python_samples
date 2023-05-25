import open3d as o3d

# 读取点云
pcd = o3d.io.read_point_cloud("point_cloud.pcd")

# 估计法向量
radius_normal = 0.1
o3d.geometry.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=30))

# 提取地面点
threshold_normal = 0.9  # 法向量与Z轴夹角的阈值
ground_plane_model, ground_plane_inliers = pcd.segment_plane(distance_threshold=0.05,
                                                         ransac_n=3,
                                                         num_iterations=1000)
ground_plane_normal = ground_plane_model[:3]
ground_plane_normal_z = abs(ground_plane_normal[-1])
ground_plane_mask = ground_plane_normal_z > threshold_normal
ground_points = pcd.select_by_index(np.where(ground_plane_mask)[0])

# 去除地面点
non_ground_points = pcd.select_by_index(np.where(~ground_plane_mask)[0])

# 显示去除地面后的点云
o3d.visualization.draw_geometries([non_ground_points])
# encoding=utf-8
