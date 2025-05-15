import pickle
with open("models/feature_list.pkl", "rb") as f:
    feature_list = pickle.load(f)
print(len(feature_list), feature_list)
