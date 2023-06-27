# Supervised machine learning algorithm
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()

# creating images and labels
images_and_labels = list(zip(digits.images, digits.target))

# to apply a classifier on this data, we need to flatten the image: instead of a 8x8 matrix we
# have to use a one-dimensional array with 64 items
data = digits.images.reshape((len(digits.images), -1))

# create a support vector machine and a support vector classifier
classifier = svm.SVC(gamma=0.001)

# split data into test and training datasets
# 75% of the original dataset is for training
train_test_split = int(len(digits.images) * 0.75)
classifier.fit(data[:train_test_split], digits.target[:train_test_split])

# now predict the value of the digit on the 25%
expected = digits.target[train_test_split:]
predicted = classifier.predict(data[train_test_split:])

print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print(accuracy_score(expected, predicted))

# let's test on the last few images
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation='nearest')
print("Prediction for the test image: ", classifier.predict(data[-2].reshape(1, -1)))

plt.show()
