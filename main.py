# Import Libraries for Project

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestClassifier
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.feature_selection import RFE
import itertools
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")

label_encoder = LabelEncoder()



#  Reading the Training Dataset¶

train = pd.read_csv("Train_data.csv")

train.info()

print(train.describe())

# Explortatory Data Analysis

ax = sns.countplot(data=train, x="protocol_class", hue="protocol_class", palette="coolwarm")

# Set tick positions and labels
ax.set_xticks([0, 1])  # Adjust tick positions according to your data
ax.set_xticklabels(["normal", "anomaly"])

# Save the plot to a file
plt.xlabel(None)
plt.savefig("OutputWithTrainingDataset/countplot.png")  # Saves the plot as countplot.png

for category in ["protocol_type", "service", "flag"]:
    if category == "service":
        plt.figure(figsize=(25, 5))
    sns.countplot(x=category, data=train, hue="protocol_class", palette=("coolwarm"))
    plt.title(f"Distribution of {category}")
    plt.xticks(rotation=45)
    plt.xlabel(None)
    plt.savefig(
        f"OutputWithTrainingDataset/Protocol_Type-{category}.png"
    )  # Saves as countplot.png in the current directory

    # Close the plot to free up memory (optional but recommended)
    plt.close()


def label_encoding(df):
    for col in df.columns:
        if df[col].dtype == "object":
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])


label_encoding(train)

train.head()


plt.figure(figsize=(40, 30))
sns.heatmap(train.corr(), annot=True, fmt=".1f", cmap="coolwarm")
plt.title("Correlations")
plt.savefig(
        "OutputWithTrainingDataset/correlations.png"
    )  # Saves as countplot.png in the current directory

    # Close the plot to free up memory (optional but recommended)
plt.close()

# Fit the encoder to your column
train["protocol_class"] = label_encoder.fit_transform(train["protocol_class"])

plt.pie(
    train["protocol_class"].value_counts(),
    labels=["normal", "anomaly"],
    autopct="%0.2f",
)
plt.savefig("OutputWithTrainingDataset/AnomolyCountPieChart.png")
plt.close()


protocolValuesCount =  train["protocol_type"].value_counts()

print(protocolValuesCount)


sns.countplot(x=train["protocol_type"])

plt.savefig(
    "OutputWithTrainingDataset/Protocol-countplot.png"
)  # Saves the plot as countplot.png



# Data Cleaning

print(train.isnull().sum())

print(print(f"Number of duplicates: {train.duplicated().sum()}"))

print(train["num_outbound_cmds"])

print(  # Drop the redundant feature
    train.drop(["num_outbound_cmds"], axis=1, inplace=True)
)

X = train.drop(["protocol_class"], axis=1)
y = train["protocol_class"]
rfc = RandomForestClassifier()

rfe = RFE(rfc, n_features_to_select=10)
rfe = rfe.fit(X, y)

feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), X.columns)]
selected_features = [v for i, v in feature_map if i == True]
top_features = pd.DataFrame({"Features": selected_features})
top_features.index = top_features.index + 1

# print(top_features)

X = X[selected_features]
scale = StandardScaler()
X = scale.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.80, random_state=2
)

# Decision Tree to Test the Data to detect Intrusion

dtc = DecisionTreeClassifier(max_depth=15, criterion="entropy")
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(
    "============================== Decision-Tree Classifier =============================="
)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Classification Report:\n", report)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("OutputWithTrainingDataset/DecisionTreeConfusionMatrix.png")


# Testing Dataset

test = pd.read_csv("Test_data.csv")

# Drop the redundant feature
test.drop(["num_outbound_cmds"], axis=1, inplace=True)


print(print(f"Number of duplicates: {test.duplicated().sum()}"))

test.drop_duplicates(inplace=True)

print(test.describe(include="object"))


# Select only the numeric columns for scaling
numeric_cols = test.select_dtypes(include=["int64", "float64"]).columns
# Ensure selected_features contains the exact features used during training
X_test = test[selected_features]  # selected_features must match those used in training


# List of non-numeric columns (e.g., 'protocol_type', 'service', etc.)
non_numeric_cols = [
    "protocol_type",
    "service",
    "flag",
]  # Replace with your actual columns

# Apply LabelEncoder to each non-numeric column
label_encoders = {}
for col in non_numeric_cols:
    encoder = LabelEncoder()
    X_test[col] = encoder.fit_transform(X_test[col])
    label_encoders[col] = encoder  # Save the encoder if needed for later use


# Apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# Ensure that all features are numeric and scaled as needed
X_test_scaled = scaler.transform(
    X_test
)  # Use the same scaler fitted on the training set

# Predict using the trained classifier
predictions = dtc.predict(X_test_scaled)

print(predictions,"dbfjdfjdj")