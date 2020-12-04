# Mushroom Classification

# Imports
import pandas as pd  # data processing
import numpy as np  # linear algebra
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # knn classifier
from sklearn.svm import SVC  # svm classifier
from sklearn import metrics  # get model perfomance
import matplotlib.pyplot as plt  # plot confusion matrix

# Import Data
data = pd.read_csv(
    "http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
    names=[  # from http://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names
        "class",
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill-attachment",
        "gill-spacing",
        "gill-size",
        "gill-color",
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
        "veil-type",
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-print-color",
        "population",
        "habitat",
    ],
)
data.head()

# Data Preparation

# Check for missing data
data.isna().any().any()


# Convert all Catergorical Data into Numerical Values (prepData)
def prepData(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def con_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1
            df[column] = list(map(con_to_int, df[column]))
    return df


prepData(data)  # apply prepData function

# Feature Definition
features = data.columns[1:]  # list all column names except 'class'
x = np.array(data[features])
y = np.array(data["class"])  # classes

# Split the training data
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=10,  # random_state set to achieve same results on other machine
)
print(
    f"""
    X Train Shape: {x_train.shape}
    Y Train Shape: {y_train.shape}
    X Test Shape: {x_test.shape}
    X Test Shape: {x_test.shape}
    """
)


# Train the Model

# Using Support Vector Machines
svm_classifier = SVC(
    random_state=10
)  # random_state set to achieve same results on other machine
svm_classifier.fit(x_train, y_train)
svm_y_pred = svm_classifier.predict(x_test)
print("SVM Prediction: ", svm_y_pred)


# Using K-Nearest Neighbours
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(x_train, y_train)
knn_y_pred = knn_classifier.predict(x_test)
print("KNN Prediction: ", knn_y_pred)


# View Model Performance (as a percentage)

print(
    f"""
    Accuracy Score\n
    SVM: {metrics.accuracy_score(y_test, svm_y_pred)*100}%\n
    KNN: {metrics.accuracy_score(y_test, knn_y_pred)*100}%\n
    -----------------------------------------------------
    """
)
print(
    f"""
    Classification Report\n
    Support Vector Machines\n
    {metrics.classification_report(y_test, svm_y_pred)}\n
    -----------------------------------------------------\n
    K-Nearest Neighbors\n
    {metrics.classification_report(y_test, knn_y_pred)}
    """
)


# Plot Confusion Matrix
metrics.plot_confusion_matrix(
    svm_classifier, x_test, y_test, display_labels=["p", "e"], cmap=plt.cm.Blues
)
plt.title("Support Vector Machines")

metrics.plot_confusion_matrix(
    knn_classifier, x_test, y_test, display_labels=["p", "e"], cmap=plt.cm.Greens
)
plt.title("K-Nearest Neighbours")
plt.show()
