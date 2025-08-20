import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from model_training import extract_features, train_model

# Загрузка и подготовка данных
def load_data(file_name="typing_data.json"):
    with open(file_name, "r", encoding="utf-8") as file:
        return json.load(file)

# Преобразование данных в X и y
def prepare_dataset(data):
    X, y = [], []
    for user, samples in data.items():
        for sample in samples:
            X.append(extract_features(sample))
            y.append(user)
    return np.array(X), np.array(y)

# Функция оценки модели
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        logloss = log_loss(y_test, y_proba)
    else:
        logloss = np.nan  # Не все модели поддерживают predict_proba

    print(f"\n{name}")
    print(f"{'-'*len(name)}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Log Loss:  {logloss:.4f}" if not np.isnan(logloss) else "Log Loss:  N/A")

# Главная функция
def compare_models():
    raw_data = load_data()
    X, y = prepare_dataset(raw_data)

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 1. Support Vector Machine
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)
    evaluate_model("Support Vector Machine (SVM)", svm, X_test, y_test)

    # 2. CatBoost (как в оригинальной реализации)
    cb = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, l2_leaf_reg=3, verbose=False)
    cb.fit(X_train, y_train)
    evaluate_model("CatBoost", cb, X_test, y_test)

if __name__ == "__main__":
    compare_models()
