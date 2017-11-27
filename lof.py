import numpy as np

# Calcule two matrices distance
def dist(A, B, d_type):
	if d_type == "Euclidean":
		return np.sqrt(np.sum(np.square(A - B), axis=1))
	else:
		return np.sum(np.absolute(A - B), axis=1)

# k-distance of p - distance between data point p and its k-th NN(nearest neigbor)
def dist_k(X, p, k, d_type, idx):
	# Remove itself
	X_ = np.delete(X, idx, axis=0)
	# Calculate distance matrix
	distance = dist(X_, p, d_type)
	# Sort
	distance.sort(axis=0)
	# Kth nearest
	return distance[k-1,0]

# k-distance neighborhood of p
def knn(X, p, k, d_type, idx):
	# Remove itself
	X_ = np.delete(X, idx, axis=0)
	# Calculate distance matrix
	distance = dist(X_, p, d_type)
	#Threshold
	t = dist_k(X, p, k, d_type, idx)
	# Filter 
	indices = np.where(distance <= t)[0]
	# Indices of points in the original dataset
	indices_o = (indices >= idx) * 1 + indices
	return np.matrix(X_[indices]), indices_o

# Reachability distance from p2 to p1:
def reachdist_k(X, p1, p2, k, d_type, idx):
	d1 = dist_k(X, p1, k, d_type, idx)
	d2 = dist(p1, p2, d_type)
	return np.maximum(d1, d2)


# Local reachability density of p
def lrd_k(X, p, k, d_type, idx):
	N, indices = knn(X, p, k, d_type, idx)
	total=0
	for i in range(N.shape[0]):
		total += reachdist_k(X, N[i], p, k, d_type, indices[i])
	return N.shape[0]/total

# Local Outlier Factor
def lof(X, p, k, d_type, idx):
	N, indices = knn(X, p, k, d_type, idx)
	reachdist=0
	lrd=0
	for i in range(N.shape[0]):
		reachdist += reachdist_k(X, N[i], p, k, d_type, indices[i])
		lrd += lrd_k(X, N[i], k, d_type, indices[i])
	return (reachdist * lrd)/np.square(N.shape[0])

# Run
def run(D, k, top_k, d_type):
	lof_m = np.zeros(D.shape[0])
	for i in range(D.shape[0]):
		lof_m[i] = lof(D, np.matrix(D[i]), k, d_type, i)
	print("For k = "+ str(k) + " with "+d_type+" distance, the top "+ str(top_k) +" outliers indices are:")
	print(lof_m.argsort()[::-1][:top_k])
	print("And their LOF values are: ")
	print(np.sort(lof_m)[::-1][:top_k])

# Load data
D = np.loadtxt(open("Q2Q3_input.csv", "rb"), delimiter=",", skiprows=1, usecols= range(1,7))
# Run it
run(D, 3, 5, "Euclidean")
run(D, 2, 5, "Manhattan")