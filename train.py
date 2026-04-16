import pandas as pd
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import extract_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import scipy.sparse as sp
import numpy as np
import joblib

print("Loading dataset3 (has raw URLs)...")
df = pd.read_csv('data/raw/dataset3.csv')
df['label'] = df['status'].apply(lambda x: 1 if x == 'phishing' else 0)
df = df[['url', 'label']].dropna()

print(f"Total rows: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")

print("\nExtracting features from URLs (this may take a minute)...")
feature_list = []
url_list     = []
labels       = []
skipped      = 0

for i, row in df.iterrows():
    feats = extract_features(row['url'])
    if feats is not None:
        feature_list.append(feats)
        url_list.append(row['url'])
        labels.append(row['label'])
    else:
        skipped += 1

print(f"Extracted: {len(feature_list)} rows, Skipped: {skipped}")

X_hand  = pd.DataFrame(feature_list)   # handcrafted numeric features
X_urls  = pd.Series(url_list)          # raw URLs for TF-IDF
y       = pd.Series(labels)

# TF-IDF on character n-grams of the URL 
# Learns patterns like 'paypal-', '-verify', '.xyz', 'login.' directly
print("\nFitting TF-IDF on URL characters...")
tfidf = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 5),   # 3-5 character sequences
    max_features=2000,    # top 2000 most informative patterns
    sublinear_tf=True,
)
X_tfidf = tfidf.fit_transform(X_urls)

# Combine handcrafted features + TF-IDF
X_hand_sparse = sp.csr_matrix(X_hand.values)
X_combined    = sp.hstack([X_hand_sparse, X_tfidf])

print(f"Combined feature matrix shape: {X_combined.shape}")

X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y, test_size=0.3, random_state=42)
X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

models = {
    'Logistic Regression': LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs'),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, max_depth=20, n_jobs=-1),
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    results[name] = {
        'model':     model,
        'accuracy':  accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall':    recall_score(y_val, y_pred),
        'f1':        f1_score(y_val, y_pred),
    }
    print(f"  Accuracy:  {results[name]['accuracy']:.4f}")
    print(f"  Precision: {results[name]['precision']:.4f}")
    print(f"  Recall:    {results[name]['recall']:.4f}")
    print(f"  F1 Score:  {results[name]['f1']:.4f}")

# Best model
best_name  = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_name]['model']
print(f"\nBest Model: {best_name} (F1: {results[best_name]['f1']:.4f})")

y_pred_test = best_model.predict(X_test)
final_acc   = accuracy_score(y_test, y_pred_test)
final_f1    = f1_score(y_test, y_pred_test)
final_prec  = precision_score(y_test, y_pred_test)
final_rec   = recall_score(y_test, y_pred_test)

print(f"\nFinal Test Results ({best_name})")
print(f"  Accuracy:  {final_acc:.4f}")
print(f"  Precision: {final_prec:.4f}")
print(f"  Recall:    {final_rec:.4f}")
print(f"  F1 Score:  {final_f1:.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred_test)}")

os.makedirs('models', exist_ok=True)

model_info = {
    'model_name': best_name,
    'accuracy':   round(final_acc  * 100, 2),
    'f1':         round(final_f1   * 100, 2),
    'precision':  round(final_prec * 100, 2),
    'recall':     round(final_rec  * 100, 2),
}

with open('models/model_info.json', 'w') as f:
    json.dump(model_info, f)

joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(tfidf,      'models/tfidf.pkl')
joblib.dump(list(X_hand.columns), 'models/feature_names.pkl')

print(f"\nSaved: best_model.pkl, tfidf.pkl, feature_names.pkl, model_info.json")
