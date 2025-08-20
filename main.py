import json
import random
import numpy as np
from handwriting import collect_typing_data
from model_training import extract_features, train_model, train_model_with_cv, save_model, load_model
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score

DATA_FILE = "typing_data.json"
MODEL_FILE = "catboost_model.cbm"
LABELS_FILE = "labels.pkl"

def load_data(file_name=DATA_FILE):
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_data(data, file_name=DATA_FILE):
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def add_new_typing_data():
    user_name = input("Введите имя пользователя: ")

    texts_pool = [
        "Сегодня 25 апреля, и погода замечательная!",
        "Уравнение 5x + 3 = 23 решается легко.",
        "Мой электронный адрес: example@mail.com.",
        "Температура воды в бассейне 26°C.",
        "Пароль от Wi-Fi: qwerty123.",
        "Общая сумма покупок: 5642 руб.",
        "Запись на приём: 18 мая в 15:00."
    ]

    texts_to_type = random.sample(texts_pool, 3)
    all_data = load_data()

    if user_name not in all_data:
        all_data[user_name] = []

    for prompt_text in texts_to_type:
        print(f"\nЗадание: {prompt_text}")
        typing_data = collect_typing_data(prompt_text)
        typing_data["prompt_text"] = prompt_text
        all_data[user_name].append(typing_data)

    save_data(all_data)
    print(f"\nВсе данные для пользователя '{user_name}' успешно сохранены.")

def recognize_user():
    all_data = load_data()
    if not all_data:
        print("Нет данных для анализа. Сначала добавьте данные.")
        return

    model = train_model(all_data)

    texts_pool = [
        "Введите PIN-код: 1234.",
        "Адрес доставки: ул. Ленина, д. 5.",
        "Код доступа: #AB12-CD34.",
        "Пожалуйста, позвоните по номеру +7-123-456-7890.",
        "Сообщение отправлено в 22:15.",
        "Скидка 15% на всё до 01.05.2025.",
        "Срок выполнения: 3 дня.",
        "Осталось 10 минут до завершения.",
        "Вам поступил перевод на сумму 3500.",
        "Не забудьте купить продукты к празднику!"
    ]

    test_text = random.choice(texts_pool)
    print(f"\nВаш текст для ввода: \"{test_text}\"")

    test_data = collect_typing_data(test_text)
    test_features = np.array([extract_features(test_data)])

    predicted_user = model.predict(test_features)[0]

    label_encoder = LabelEncoder()
    user_classes = list(all_data.keys())
    label_encoder.fit(user_classes)

    test_proba = model.predict_proba(test_features)[0]
    predicted_index = label_encoder.transform([predicted_user])[0]

    logloss = log_loss(
        [predicted_index],
        [test_proba],
        labels=list(range(len(user_classes)))
    )

    print(f"Вероятности для каждого пользователя:")
    for user, prob in zip(label_encoder.classes_, test_proba):
        print(f"  {user}: {prob:.4f}")

    print(f"Предполагаемый пользователь: {predicted_user}")

    while True:
        agree = input("Вы согласны с предсказанием? (да/нет): ").lower()
        if agree == "да":
            true_label = label_encoder.transform([predicted_user])
            logloss = log_loss(true_label, [test_proba], labels=list(range(len(user_classes))))
            print("Отлично! Предсказание подтверждено.")
            print(f"Log Loss: {logloss:.4f}")
            break
        elif agree == "нет":
            print("Предсказание не принято. Log Loss не рассчитывается.")
            break
        else :
            print("Неверный ввод. Пожалуйста, введите 'да' или 'нет'.")

def main():
    print("Выберите действие:")
    print("1. Добавить новые данные о почерке пользователя.")
    print("2. Определить пользователя по его почерку.")
    choice = input("Введите номер действия: ")

    while True:
        if choice == "1":
            add_new_typing_data()
            break
        elif choice == "2":
            recognize_user()
            break
        else:
            print("Неверный выбор действия.")

if __name__ == "__main__":
    main()

