# Un-Supervised machine learning algorithm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples=100, centers=3, random_state=0, cluster_std=3)

plt.scatter(x[:, 0], x[:, 1], s=50)

model = KMeans(3)
model.fit(x)
y_kmeans = model.predict(x)
print(y_kmeans)

plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=50, cmap='rainbow')
plt.show()
