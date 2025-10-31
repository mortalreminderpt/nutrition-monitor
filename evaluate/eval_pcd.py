# should use a new python3.12 environment to run (different to pyproject.toml)
import sys
import open3d as o3d
import numpy as np

# Best sample
color = o3d.io.read_image("comp-90086-nutrition-5-k/Nutrition5K/Nutrition5K/train/color/dish_0747/rgb.png")
depth = o3d.io.read_image("comp-90086-nutrition-5-k/Nutrition5K/Nutrition5K/train/depth_raw/dish_0747/depth_raw.png")

# Worst sample
# color = o3d.io.read_image("comp-90086-nutrition-5-k/Nutrition5K/Nutrition5K/train/color/dish_0783/rgb.png")
# depth = o3d.io.read_image("comp-90086-nutrition-5-k/Nutrition5K/Nutrition5K/train/depth_raw/dish_0783/depth_raw.png")

h, w = np.asarray(color).shape[:2]

# 1m = 10000 unit and truncate at 0.4m
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth, depth_scale=10000.0, depth_trunc=0.4, convert_rgb_to_intensity=False
)

fx = fy = float(max(w, h))
cx, cy = w / 2.0, h / 2.0
intr = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)

pcd.transform([[1,0,0,0],
               [0,-1,0,0],
               [0,0,-1,0],
               [0,0,0,1]])

# rotate pcd to best view
R = o3d.geometry.get_rotation_matrix_from_xyz((np.deg2rad(-60), np.deg2rad(-2), np.deg2rad(120)))
pcd.rotate(R, center=(0, 0, 0))

o3d.io.write_point_cloud("view.ply", pcd)
o3d.visualization.draw_geometries([pcd])