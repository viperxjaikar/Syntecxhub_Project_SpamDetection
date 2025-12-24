import joblib

model = joblib.load("spam_model.pkl")

samples = [
    "Free offer available today, let me know",
    "Can we meet at 6 pm tomorrow?",
    "Urgent call me when you get this message",
    "Congratulations you are selected for a reward"
]

preds = model.predict(samples)

for text, p in zip(samples, preds):
    label = "spam" if p == 1 else "ham"
    print(f"{label} -> {text}")
