# --- Import librării ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# --- Creare dataset fictiv emailuri ---
# ============================================================
data = {
    "Email": [
        "Câștigă acum 1000$ printr-un click!",
        "Programare întâlnire luni la 10",
        "Ofertă limitată la medicamente online",
        "Întâlnire echipă proiect vineri",
        "Ai fost selectat pentru un premiu gratuit",
        "Rezumatul proiectului din această săptămână",
        "Reduceri uriașe doar azi la electronice",
        "Confirmare rezervare hotel",
        "Felicitări! Ai câștigat un iPhone!",
        "Raport vânzări luna septembrie",
        "Oferte exclusive pentru tine, nu rata!",
        "Invitație la webinar de marketing",
        "Cumpără acum și primești cadou!",
        "Plan ședință echipă luni dimineața",
        "Alertă securitate cont: verifică linkul",
        "Notificare factură lunară",
        "Reducere 50% la toate abonamentele!",
        "Agenda întâlnirilor săptămânii",
        "Premiu instant la concursul nostru online!",
        "Actualizare proiect: sarcini finalizate"
    ],
    "Label": [
        "SPAM",
        "HAM",
        "SPAM",
        "HAM",
        "SPAM",
        "HAM",
        "SPAM",
        "HAM",
        "SPAM",
        "HAM",
        "SPAM",
        "HAM",
        "SPAM",
        "HAM",
        "SPAM",
        "HAM",
        "SPAM",
        "HAM",
        "SPAM",
        "HAM"
    ]
}

df = pd.DataFrame(data)

# ============================================================
# --- Mapare clase la valori binare ---
# ============================================================
# 0 = NOT_SPAM (HAM), 1 = SPAM
df["Label"] = df["Label"].map({"HAM": 0, "SPAM": 1})
y = df["Label"]

# ============================================================
# --- Vectorizare emailuri ---
# ============================================================
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Email"])

# ============================================================
# --- Împărțire train/test ---
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ============================================================
# --- Antrenare Logistic Regression ---
# ============================================================
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ============================================================
# --- Evaluare ---
# ============================================================

def evaluate_seaborn(y_true, y_pred, model_name="Model"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

evaluate_seaborn(y_test, y_pred, "Logistic Regression Email SPAM")


# def evaluate(y_true, y_pred, model_name="Model"):
#     print(f"\n--- Evaluare {model_name} ---")
#     print("Accuracy:", accuracy_score(y_true, y_pred))
#     print("Precision (SPAM):", precision_score(y_true, y_pred))
#     print("Recall (SPAM):", recall_score(y_true, y_pred))
#
#     cm = confusion_matrix(y_true, y_pred)
#     print("Confusion Matrix:\n", cm)
#
#     plt.matshow(cm, cmap=plt.cm.Blues)
#     plt.title(f"Confusion Matrix - {model_name}")
#     plt.colorbar()
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
#     plt.show()
#
# evaluate(y_test, y_pred, "Logistic Regression Email SPAM")
