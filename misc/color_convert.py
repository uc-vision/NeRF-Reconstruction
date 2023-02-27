import numpy as np
import open3d as o3d

fp_in = "./nerf/outputs/plant_and_food_30000_050.pcd"
fp_out = "./nerf/outputs/plant_and_food_30000_050_rgb.pcd"
 
pointcloud = o3d.io.read_point_cloud(fp_in)

colors = np.asarray(pointcloud.colors)

colors = colors[:, ::-1]

pointcloud.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud(fp_out, pointcloud)