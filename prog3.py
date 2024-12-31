import pandas as pd
import numpy as np

dataset = pd.read_csv('playtennis.csv', names=['outlook', 'temperature', 'humidity', 'wind', 'class'])

def entropy(col):
    _, counts = np.unique(col, return_counts=True)
    return -sum((count / sum(counts)) * np.log2(count / sum(counts)) for count in counts)

def info_gain(data, feature, target="class"):
    total_entropy = entropy(data[target])
    vals, counts = np.unique(data[feature], return_counts=True)
    return total_entropy - sum((counts[i] / sum(counts)) * entropy(data[data[feature] == vals[i]][target]) for i in range(len(vals)))

def ID3(data, features, target="class"):
    if len(np.unique(data[target])) == 1: return data[target].iloc[0]
    if not features: return data[target].mode()[0]
    best_feature = max(features, key=lambda f: info_gain(data, f, target))
    return {best_feature: {v: ID3(data[data[best_feature] == v], [f for f in features if f != best_feature], target) for v in np.unique(data[best_feature])}}

def predict(query, tree):
    for feature, branches in tree.items():
        return predict(query, branches.get(query.get(feature))) if isinstance(branches.get(query.get(feature)), dict) else branches.get(query.get(feature))

queries = dataset.iloc[:14, :-1].to_dict(orient="records")
tree = ID3(dataset.iloc[:14], dataset.columns[:-1].tolist())
accuracy = sum(predict(q, tree) == t for q, t in zip(queries, dataset["class"][:14])) / 14 * 100
print("Tree:", tree, "\nAccuracy:", accuracy, "%")
