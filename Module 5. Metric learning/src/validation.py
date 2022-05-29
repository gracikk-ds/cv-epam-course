from scipy import spatial
import time
import pickle
import numpy as np
from sklearn.metrics import classification_report

with open("../data/results/embeddings_train.pickle", "rb") as handle:
    data_train = pickle.load(handle)
    classes_train = np.array(data_train[1])
    embeddings_train = data_train[2]

with open("../data/results/embeddings_test.pickle", "rb") as handle:
    data_test = pickle.load(handle)
    classes_test = np.array(data_test[1], dtype=np.uint8)
    embeddings_test = data_test[2]

with open("../logs/accuracy.pickle", "rb") as handle:
    acc = pickle.load(handle)
    print(acc)

start = time.time()

tree = spatial.KDTree(embeddings_train)
_, idx = tree.query(embeddings_test)
predictions = np.array(classes_train[idx], dtype=np.uint8)

print(classification_report(classes_test, predictions))
