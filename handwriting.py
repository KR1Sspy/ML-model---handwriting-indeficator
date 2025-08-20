import time
from pynput import keyboard

def collect_typing_data(prompt_text):
    user_typing_data = {
        "key_press_durations": [],
        "key_intervals": [],
        "total_typing_time": 0,
        "typed_text": "",
        "accuracy": 0,
        "backspace_count": 0
    }

    key_press_times = {}
    last_key_time = None
    typing_start_time = None

    def on_press(key):
        nonlocal last_key_time, typing_start_time

        try:
            if typing_start_time is None:
                typing_start_time = time.time()

            if key == keyboard.Key.backspace:
                user_typing_data["backspace_count"] += 1

            if hasattr(key, 'char') and key.char:
                user_typing_data["typed_text"] += key.char
        except AttributeError:
            pass

        key_press_times[key] = time.time()

        if last_key_time is not None:
            interval = key_press_times[key] - last_key_time
            user_typing_data["key_intervals"].append(interval)

        last_key_time = key_press_times[key]

    def on_release(key):
        try:
            if key in key_press_times:
                duration = time.time() - key_press_times[key]
                user_typing_data["key_press_durations"].append(duration)
                del key_press_times[key]
        except KeyError:
            pass

        if key == keyboard.Key.esc:
            user_typing_data["total_typing_time"] = time.time() - typing_start_time
            return False

    print(f"Введите текст: \"{prompt_text}\". Для завершения нажмите ESC...")
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    def calculate_accuracy(typed_text, prompt_text):
        correct_chars = sum(1 for a, b in zip(typed_text, prompt_text) if a == b)
        accuracy = correct_chars / len(prompt_text) if len(prompt_text) > 0 else 0
        return round(accuracy * 100, 2)

    user_typing_data["accuracy"] = calculate_accuracy(user_typing_data["typed_text"], prompt_text)
    return user_typing_data
