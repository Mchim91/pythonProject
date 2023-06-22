from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import datasets

digit_data = datasets.load_digits()

# Convert to a one dimensional array
image_features = digit_data.images.reshape((len(digit_data.images), -1))
image_targets = digit_data.target

random_forest_model = RandomForestClassifier()
