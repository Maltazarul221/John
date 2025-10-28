# --- Import librării ---
import zipfile
from io import BytesIO

import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt

# URL dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'

# Download fișierul ZIP
response = requests.get(url)
zip_data = BytesIO(response.content)

# Deschide ZIP și citește fișierul SMSSpamCollection
with zipfile.ZipFile(zip_data) as z:
    with z.open('SMSSpamCollection') as f:
        df = pd.read_csv(f, sep='\t', header=None, names=['Label', 'Message'])

# Verificăm primele 5 rânduri
print(df.head(20))

# --- Împărțire train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    df['Message'], df['Label'], test_size=0.3, random_state=42, stratify=df['Label']
)
