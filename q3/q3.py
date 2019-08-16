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
		
		i1 = points[i]["image"][0]
		i2 = points[i]["image"][1]
		
		row1 = [-w1,-w2,-1,0,0,0,i1*w1,i1*w2,i1]
		row2 = [0,0,0,-w1,-w2,-1,i2*w1,i2*w2,i2]

		matrix.append(row1)
		matrix.append(row2)

	return matrix

def get_H_matrix(M_matrix):
	u, s, vh = np.linalg.svd(M_matrix, full_matrices=False)

	vh = vh.transpose()

	# print(M_matrix)
	# print(np.matmul(u, np.matmul(s, vh.transpose())))
	P_vec = []
	for i in range(9):
		P_vec.append(vh[i][8])

	H_matrix = []
	H_matrix.append(P_vec[0:3])
	H_matrix.append(P_vec[3:6])
	H_matrix.append(P_vec[6:9])
	return np.array(H_matrix)

def get_R_T_matrix(H_matrix, K):
	K_inv = np.linalg.inv(K)	
	
	R = np.matmul(K_inv, H_matrix)
	# print(R)

	R_temp = np.ones((3,3))
	R_temp[:, 0] = R[:, 0]
	R_temp[:, 1] = R[:, 1]

	T = R[:, 2]/np.linalg.norm(R[:, 0])

	R_temp[:, 2] = np.cross(R[:, 0], R[:, 1])

	u, s, vh = np.linalg.svd(R_temp, full_matrices=False)

	determinant = np.linalg.det(np.matmul(u, vh.transpose()))

	s = np.zeros((3,3))
	s[0][0] = 1
	s[1][1] = 1
	s[2][2] = determinant
	R = np.matmul(u, np.matmul(s, vh))
	
	return R, T

K = np.array([[406.952636, 0.000000, 366.184147], [0.000000, 405.671292, 244.705127], [0.000000, 0.000000, 1.000000]])

total, points = correspondences3D2D()

world_points = []
for i in range(total):
	rec = [points[i]["world"][0], points[i]["world"][1], 1]
	world_points.append(rec)
world_points = np.array(world_points)

M_matrix = calibrateDLT(total, points)
H_matrix = get_H_matrix(M_matrix)
R, T = get_R_T_matrix(H_matrix, K)

T = np.array([T])
P = np.concatenate((R, T.T), axis=1)
print(P)

# test = np.matmul(K, np.matmul(P, [0.2105, 0, 0, 1])) 
# print(test/test[2])

# transform = np.matmul(K, P)
# print(transform)
# print(H_matrix)

new_points = np.matmul(H_matrix, world_points.T).T

for i in range(total):
	new_points[i] /= new_points[i][2]
# print(new_points)

img = mpimg.imread('image.png')
plt.imshow(img)

plt.scatter(x = new_points[:, 0], y = new_points[:, 1], c='r', s=10)
plt.show()
# print(H_matrix)


	
# print(H_matrix[:, 0])
# test = np.matmul(H_matrix, np.array([0, 0.1315, 1])) 
# print(test/test[2])