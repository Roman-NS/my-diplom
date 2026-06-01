import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import defaultdict
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')


def parse_logs_with_drain_optimized(log_file):
    config = TemplateMinerConfig()
    config.sim_th = 0.5
    config.depth = 4
    parser = TemplateMiner(config=config)

    block_data = defaultdict(lambda: {'templates': [], 'event_ids': [], 'raw_count': 0})

    line_count = 0
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            result = parser.add_log_message(line)
            block_match = re.search(r'blk_([-\d]+)', line)
            block_id = block_match.group(0) if block_match else "NO_BLOCK"

            block_data[block_id]['templates'].append(result["template_mined"])
            block_data[block_id]['event_ids'].append(result["cluster_id"])
            block_data[block_id]['raw_count'] += 1

            line_count += 1
            if line_count % 2_000_000 == 0:
                print(f"Обработано строк: {line_count:,}")

    print(f"\nПарсинг завершён. Блоков: {len(block_data):,}")

    data = []
    for block_id, info in block_data.items():
        data.append({
            'block_id': block_id,
            'template': ' || '.join(info['templates'][:60]),
            'event_ids': info['event_ids'],
            'event_count': info['raw_count']
        })
    return pd.DataFrame(data)


if __name__ == "__main__":
    LOG_FILE = "HDFS.log"
    LABEL_FILE = "anomaly_label.csv"

    df_blocks = parse_logs_with_drain_optimized(LOG_FILE)

    labels = pd.read_csv(LABEL_FILE)
    if 'BlockId' in labels.columns:
        labels = labels.rename(columns={'BlockId': 'block_id'})
    if 'Label' in labels.columns:
        labels = labels.rename(columns={'Label': 'label'})

    df = df_blocks.merge(labels, on='block_id', how='inner')
    df['label'] = df['label'].map({'Normal': 0, 'Anomaly': 1}).fillna(0)

    print("\nРаспределение классов:\n", df['label'].value_counts())

    normal_df = df[df['label'] == 0].copy()

    print("\nОбучение Isolation Forest...")
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), dtype=np.float32)

    sample_size = min(100000, len(normal_df))
    normal_sample = normal_df.sample(n=sample_size, random_state=42)

    X_normal = vectorizer.fit_transform(normal_sample['template'])
    scaler = StandardScaler(with_mean=False)
    X_normal_scaled = scaler.fit_transform(X_normal)

    iforest = IsolationForest(contamination=0.02, n_estimators=200,
                              max_samples=30000, random_state=42, n_jobs=-1)
    iforest.fit(X_normal_scaled)

    X_all = vectorizer.transform(df['template'])
    X_all_scaled = scaler.transform(X_all)
    if_pred = (iforest.predict(X_all_scaled) == -1).astype(int)

    print("\n" + "=" * 70)
    print("                    РЕЗУЛЬТАТЫ МОДЕЛИ")
    print("=" * 70)
    print(classification_report(df['label'], if_pred))

    comparison = pd.DataFrame({
        'Model': ['Isolation Forest (TF-IDF)'],
        'F1_Anomaly': [f1_score(df['label'], if_pred)],
        'ROC_AUC': [roc_auc_score(df['label'], -iforest.decision_function(X_all_scaled))]
    })
    print(comparison.round(4))

    # График
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(df['label'], if_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Isolation Forest')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    comparison.to_csv("model_comparison.csv", index=False)
    joblib.dump((iforest, vectorizer, scaler), "iforest_model.pkl")
    print("\nМодель сохранена!")