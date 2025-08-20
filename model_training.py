import numpy as np
from collections import Counter
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pickle

def extract_features(typing_data, max_key_press=50, max_key_interval=50):
    key_press_durations = typing_data["key_press_durations"][:max_key_press]
    key_intervals = typing_data["key_intervals"][:max_key_interval]

    key_press_durations += [0] * (max_key_press - len(key_press_durations))# дополнение нулями пропущенных данных
    key_intervals += [0] * (max_key_interval - len(key_intervals))

    accuracy = typing_data.get("accuracy", 0)# обработка шумов
    backspace_count = typing_data.get("backspace_count", 0)

    features = key_press_durations + key_intervals + [typing_data["total_typing_time"], accuracy, backspace_count]
    return features

def train_model(data, max_key_press=50, max_key_interval=50):
    X, y = [], []

    for user_name, samples in data.items():
        for sample in samples:
            X.append(extract_features(sample, max_key_press, max_key_interval))
            y.append(user_name)

    X = np.array(X)
    y = np.array(y).ravel()

    class_counts = Counter(y)
    total_samples = len(y)
    class_weights = {class_name: total_samples / count for class_name, count in class_counts.items()}# учитывается для каждого класса (важность)

    weight_list = [class_weights[class_name] for class_name in sorted(class_counts.keys())]

    model = CatBoostClassifier(iterations=300,
                               depth=6,
                               learning_rate=0.05,
                               l2_leaf_reg=3,
                               class_weights=weight_list,
                               verbose=False)
    model.fit(X, y)
    return model

def train_model_with_cv(data, max_key_press=50, max_key_interval=50, n_splits=5):
    X, y = [], []
    for user_name, samples in data.items():
        for sample in samples:
            X.append(extract_features(sample, max_key_press, max_key_interval))
            y.append(user_name)

    X = np.array(X)
    y = np.array(y).ravel()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        class_counts = Counter(y_train)
        total_samples = len(y_train)
        class_weights = {class_name: total_samples / count for class_name, count in class_counts.items()}
        weight_list = [class_weights[class_name] for class_name in sorted(class_counts.keys())]

        model = CatBoostClassifier(iterations=300,
                                   depth=6,
                                   learning_rate=0.05,
                                   l2_leaf_reg=3,
                                   class_weights=weight_list,
                                   verbose=False)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        accuracies.append(acc)

    # Обучаем финальную модель на всех данных
    final_class_counts = Counter(y)
    final_class_weights = {cls: len(y) / count for cls, count in final_class_counts.items()}
    final_weight_list = [final_class_weights[cls] for cls in sorted(final_class_counts.keys())]

    final_model = CatBoostClassifier(iterations=300,
                                     depth=6,
                                     learning_rate=0.05,
                                     l2_leaf_reg=3,
                                     class_weights=final_weight_list,
                                     verbose=False)
    final_model.fit(X, y)

    class_labels = sorted(final_class_counts.keys())
    return final_model, class_labels

def save_model(model, class_labels, model_path, labels_path):
    model.save_model(model_path)
    with open(labels_path, "wb") as f:
        pickle.dump(class_labels, f)

def load_model(model_path, labels_path):
    model = CatBoostClassifier()
    model.load_model(model_path)
    with open(labels_path, "rb") as f:
        class_labels = pickle.load(f)
    return model, class_labels
