import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np


def correspondences3D2D():
	points = []
	total = int(input())
	
	for i in range(total):
		image_point = [float(x) for x in input().split()]
		world_point = [float(x) for x in input().split()]
		dictionary = {
			"image" : image_point,
			"world" : world_point,
		}
		points.append(dictionary)

	return total, points

def calibrateDLT(total, points):
	matrix = []
	for i in range(total):
		w1 = points[i]["world"][0]
		w2 = points[i]["world"][1]
		w3 = points[i]["world"][2]

		i1 = points[i]["image"][0]
		i2 = points[i]["image"][1]
		row1 = [-w1,-w2,-w3,-1,0,0,0,0,i1*w1,i1*w2,i1*w3,i1]
		row2 = [0,0,0,0,-w1,-w2,-w3,-1,i2*w1,i2*w2,i2*w3,i2]
		matrix.append(row1)
		matrix.append(row2)

	# print(matrix)

	u, s, vh = np.linalg.svd(matrix, full_matrices=False)
	# print(vh)

	vh = vh.transpose()
	P_vec = []
	for i in range(12):
		P_vec.append(vh[i][11])

	# print(P_vec)

	P_mat = []
	P_mat.append(P_vec[0:4])
	P_mat.append(P_vec[4:8])
	P_mat.append(P_vec[8:12])
	return P_mat

w1 = np.array([0, 1.51, 4.1, 1])
w2 = np.array([1.38, 1.51, 4.1, 1])

total, points = correspondences3D2D()
P_mat = calibrateDLT(total,points)

p1 = np.matmul(P_mat, w1)
p1 = p1/p1[2]

p2 = np.matmul(P_mat, w2)
p2 = p2/p2[2]

x = [750, 750, 827, 953, 827, 953, p1[0], p2[0]]
y = [184, 264, 190, 190, 303, 303, p1[1], p2[1]]

# Read Images 
img = mpimg.imread('image.png') 
  
# Output Images 
plt.imshow(img) 
plt.scatter(x=x, y=y, c='b', s=30, zorder = 2)

points = [[0,1], [0,2], [1,4], [2,4], [3,7], [6,7], [5,6], [3,5], [2,3], [4,5], [0,7], [1,6]]

for p in points:
	i = p[0]
	j = p[1]
	plt.plot([x[i], x[j]],[y[i], y[j]], c='r', linewidth=4, zorder = 1)

plt.show()
# temp = np.matmul(P_mat, np.array([1.38, 0, 4.1, 1]))
# print(temp/ temp[2])