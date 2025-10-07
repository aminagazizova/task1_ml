from transformers import pipeline

analyzer = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

texts = [
    "Этот проект отличный!",
    "Мне совсем не понравилось выполнение задания.",
    "Результат нормальный, но можно улучшить.",
    "Просто ужасно — я не доволен.",
    "Замечательная работа! Всё получилось."
]

print("\nРезультаты анализа тональности:\n")
results = analyzer(texts)

for text, res in zip(texts, results):
    label = res['label']
    score = res['score']
    print(f"{text}\n→ {label} ({score:.2f})\n")
