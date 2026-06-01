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

import warnings

warnings.filterwarnings('ignore')


def parse_logs_with_drain_optimized(log_file):

    config = TemplateMinerConfig()
    config.sim_th = 0.5
    config.depth = 4

    parser = TemplateMiner(config=config)

    block_data = defaultdict(lambda: {
        'templates': [],
        'event_ids': [],
        'raw_count': 0
    })

    line_count = 0

    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            result = parser.add_log_message(line)

            block_match = re.search(r'blk_([-\d]+)', line)

            block_id = (
                block_match.group(0)
                if block_match
                else f"NO_BLOCK_{line_count}"
            )

            block_data[block_id]['templates'].append(
                result["template_mined"]
            )

            block_data[block_id]['event_ids'].append(
                result["cluster_id"]
            )

            block_data[block_id]['raw_count'] += 1

            line_count += 1

            if line_count % 2_000_000 == 0:
                print(f"Обработано строк: {line_count:,}")

    print(f"\nПарсинг завершён.")
    print(f"Всего блоков: {len(block_data):,}")

    data = []

    for block_id, info in block_data.items():

        data.append({
            'block_id': block_id,
            'template': ' || '.join(info['templates'][:60]),
            'event_count': info['raw_count']
        })

    return pd.DataFrame(data)


if __name__ == "__main__":

    LOG_FILE = "hdfs.log"

    print("\nЗагрузка модели...")

    iforest, vectorizer, scaler = joblib.load(
        "iforest_model.pkl"
    )

    print("Модель загружена.")

    print("\nПарсинг логов...")

    df = parse_logs_with_drain_optimized(LOG_FILE)

    print("\nПреобразование логов...")

    X = vectorizer.transform(df['template'])

    X_scaled = scaler.transform(X)


    print("\nАнализ аномалий...")

    predictions = iforest.predict(X_scaled)

    # -1 = anomaly
    #  1 = normal

    df['anomaly'] = np.where(
        predictions == -1,
        1,
        0
    )


    df['anomaly_score'] = -iforest.decision_function(
        X_scaled
    )


    total_logs = len(df)

    anomaly_count = df['anomaly'].sum()

    normal_count = total_logs - anomaly_count

    print("\n" + "=" * 70)
    print("               РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("=" * 70)

    print(f"Всего блоков: {total_logs:,}")

    print(f"Аномалий найдено: {anomaly_count:,}")

    print(f"Нормальных блоков: {normal_count:,}")

    print(
        f"Процент аномалий: "
        f"{(anomaly_count / total_logs) * 100:.2f}%"
    )


    anomalies = df[df['anomaly'] == 1].copy()

    anomalies = anomalies.sort_values(
        by='anomaly_score',
        ascending=False
    )

    print("\nТоп-10 наиболее аномальных блоков:\n")

    print(
        anomalies[
            [
                'block_id',
                'anomaly_score',
                'event_count'
            ]
        ].head(10)
    )


    df.to_csv(
        "anomaly_detection_results.csv",
        index=False
    )

    anomalies.to_csv(
        "detected_anomalies.csv",
        index=False
    )

    print("\nРезультаты сохранены:")

    print(" - anomaly_detection_results.csv")

    print(" - detected_anomalies.csv")


    plt.figure(figsize=(7, 5))

    sns.countplot(
        x=df['anomaly']
    )

    plt.title(
        'Distribution of Detected Anomalies'
    )

    plt.xlabel(
        'Class (0 = Normal, 1 = Anomaly)'
    )

    plt.ylabel(
        'Count'
    )

    plt.tight_layout()

    plt.show()

    print("\nАнализ завершён.")