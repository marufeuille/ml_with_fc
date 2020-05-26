import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression

df = pd.read_csv("../data/creditcard.csv")

# Train test split
pos_df = df[df["Class"] == 1]
neg_df = df[df["Class"] == 0]
np.random.seed(5)

pos_df["group"] = np.random.randint(0,10, size=pos_df.shape[0])
neg_df["group"] = np.random.randint(0,10, size=neg_df.shape[0])

train = pd.concat([
    neg_df[neg_df["group"] < 8],
    pos_df[pos_df["group"] < 8],
]).sample(frac=1).reset_index(drop=True).drop(columns=["group"])

test = pd.concat([
    neg_df[neg_df["group"] >= 8],
    pos_df[pos_df["group"] >= 8],
]).sample(frac=1).reset_index(drop=True).drop(columns=["group"])

train.to_csv("../data/creditcard_train.csv")
test.to_csv("../data/creditcard_test.csv")

X_train = train.drop(columns=["Class"])
X_test = test.drop(columns=["Class"])
y_train = train["Class"]
y_test = test["Class"]

# learning
lr = LogisticRegression(
    random_state=0,
    penalty="l2",
    C=1.0,
    solver='liblinear'
).fit(X_train, y_train)

with open("../model/creditcard_fraud_detection_model.pkl", "wb") as f:
    pickle.dump(lr, f)