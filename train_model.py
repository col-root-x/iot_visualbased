import json
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def extract_windows(filename, label, window_size=30):
    sequences = defaultdict(list)
    X, y = [], []
    
    with open(filename, "r") as file:
        for line in file:
            data = json.loads(line)
            # Flatten the 17 [x,y] coordinates into a 34-feature array
            flat_kpts = np.array(data["keypoints"]).flatten()
            sequences[data["track_id"]].append(flat_kpts)
            
    for track_id, frames in sequences.items():
        if len(frames) >= window_size:
            for i in range(len(frames) - window_size + 1):
                window = frames[i : i + window_size]
                X.append(np.concatenate(window)) # 1,020 features per window
                y.append(label)
                
    return X, y

print("Processing data...")
X_normal, y_normal = extract_windows("normal_behavior.jsonl", 0)
X_fight, y_fight = extract_windows("fighting_behavior.jsonl", 1)

# Combine and split the data
X = np.array(X_normal + X_fight)
y = np.array(y_normal + y_fight)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Random Forest on {len(X_train)} samples...")
clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model weights
joblib.dump(clf, "fight_classifier.pkl")
print("Saved model as 'fight_classifier.pkl'")
