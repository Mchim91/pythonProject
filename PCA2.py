# Supervised machine learning algorithm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

mnist_data = fetch_openml(name='mnist_784', version=1, parser='auto')

features = mnist_data.data
targets = mnist_data.target

train_img, test_img, train_lbl, test_lbl = train_test_split(features, targets, test_size=0.15, random_state=0)

scalar = StandardScaler()
# The mean and standard deviation is carried out on the training image
scalar.fit(train_img)
# the z-transformation is carried out on the training and testing images
train_img = scalar.transform(train_img)
test_img = scalar.transform(test_img)

print(train_img.shape)

# we keep 95% variance - so 95% of the original information
pca = PCA(.95)
pca.fit(train_img)

train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

print(train_img.shape)


