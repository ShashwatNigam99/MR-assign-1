import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np

# [0, 0, 0] (World Frame) - > [827, 310]
def get_coord():
	points = []
	total = int(input())
	
	for i in range(total):
		world_point = [float(x) for x in input().split()]
		points.append(np.array(world_point))

	return total, points

K = np.array([[7.2153e+02,0,6.0955e+02], [0,7.2153e+02,1.7285e+02], [0,0,1]])

f_x = K[0][0]
f_y = K[1][1]
o_x = K[0][2]
o_y = K[1][2]

Y_cam = 1.65

Z_cam = (f_y * Y_cam)/ (310-o_y)
X_cam = ((827-o_x) * Z_cam)/ f_x

print(X_cam, Y_cam, Z_cam)

total, points = get_coord()

ref_point = np.array([X_cam, Y_cam, Z_cam])
in_camera_frame = []

for i in range(total):
	in_camera_frame.append(ref_point + points[i])
in_camera_frame = np.array(in_camera_frame)

image_frame = np.matmul(K, in_camera_frame.T).T

plot_points = []
for i in range(total):
	plot_points.append(image_frame[i]/image_frame[i][2])

plot_points = np.array(plot_points)

theta = np.radians(5)
rotation_mat = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

plot_points = np.matmul(rotation_mat, plot_points.T).T

points = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]]

img = mpimg.imread('image.png') 

plt.scatter(x=plot_points[:,0], y=plot_points[:,1], c='b', s=30, zorder = 2, marker='x')

# Output Images 
plt.imshow(img) 

for p in points:
	i = p[0]
	j = p[1]

	x1, x2 = plot_points[i][0], plot_points[j][0] 
	y1, y2 = plot_points[i][1], plot_points[j][1] 
	plt.plot([x1, x2], [y1, y2], c='r', linewidth=4, zorder = 1)

plt.show()

