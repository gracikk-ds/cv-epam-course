import pickle
import numpy as np
from sklearn.metrics import classification_report, pairwise_distances

with open("../data/results/embeddings_train.pickle", "rb") as handle:
    data_train = pickle.load(handle)
    classes_train = np.array(data_train[1])
    embeddings_train = np.array(data_train[2])
    print(embeddings_train.shape)

with open("../data/results/embeddings_test.pickle", "rb") as handle:
    data_test = pickle.load(handle)
    classes_test = np.array(data_test[1], dtype=np.uint8)
    embeddings_test = np.array(data_test[2])
    print(embeddings_test.shape)

dist = pairwise_distances(embeddings_test, embeddings_train, metric="cosine")
predictions = np.array(classes_train[dist.argmin(axis=1)], dtype=np.uint8)
print(classification_report(classes_test, predictions))
