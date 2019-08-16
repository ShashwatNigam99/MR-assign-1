from load_points import load_velodyne_points
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 


points = load_velodyne_points('lidar-points.bin')

K = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02], [0.000000e+00, 7.215377e+02, 1.728540e+02], [0.000000e+00, 0.000000e+00, 1.000000e+00]])
T = np.array([[0, 0, 1, 0.27], [-1, 0, 0, 0.06], [0, -1, 0, -0.08], [0, 0, 0, 1]])
T = np.linalg.inv(T)
T = T[:3, :]

newrow = [1] * points.shape[0]

newrow = np.array([newrow])
points = np.concatenate((points, newrow.T), axis=1)

P = np.matmul(K, T)
transformed_points = np.matmul(P, points.T)

new_points = transformed_points.T

for i in range(len(new_points)):
	new_points[i] /= new_points[i][2]

to_plot = []
depth = []

i = 0
for i in range(len(new_points)):
	if 0 <= new_points[i][0] <= 1242 and 0 <= new_points[i][1] <= 375:
		to_plot.append(new_points[i]) 
		depth.append(points[i][0])
	i += 1

to_plot = np.array(to_plot)
depth = np.array(depth)

img = mpimg.imread('image.png')
  
# Output Images 
plt.imshow(img) 
plt.scatter(x=to_plot[:, 0], y=to_plot[:, 1], c=depth*10, cmap='rainbow', s=1)

plt.show()