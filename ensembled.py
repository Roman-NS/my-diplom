import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import psutil
import warnings
import joblib
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings('ignore')


def get_memory_usage_mb():

    process = psutil.Process(os.getpid())

    return process.memory_info().rss / 1024 / 1024


def parse_logs(log_file="HDFS.log"):

    print("\nПарсинг логов...")

    config = TemplateMinerConfig()
    config.sim_th = 0.5
    config.depth = 4

    parser = TemplateMiner(config=config)

    block_data = defaultdict(
        lambda: {
            'templates': [],
            'event_ids': [],
            'raw_count': 0
        }
    )

    with open(
        log_file,
        'r',
        encoding='utf-8',
        errors='ignore'
    ) as f:

        for line_num, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            result = parser.add_log_message(line)

            block_match = re.search(
                r'blk_([-\d]+)',
                line
            )

            block_id = (
                block_match.group(0)
                if block_match
                else "NO_BLOCK"
            )

            block_data[block_id]['templates'].append(
                result["template_mined"]
            )

            block_data[block_id]['event_ids'].append(
                result["cluster_id"]
            )

            block_data[block_id]['raw_count'] += 1

            if line_num % 2_000_000 == 0:

                print(
                    f"Обработано строк: "
                    f"{line_num:,}"
                )

    data = []

    for block_id, info in block_data.items():

        data.append({

            'block_id': block_id,

            'template': ' || '.join(
                info['templates'][:60]
            ),

            'event_ids': info['event_ids'],

            'event_count': info['raw_count']
        })

    df = pd.DataFrame(data)

    print(
        f"\nПарсинг завершён. "
        f"Блоков: {len(df):,}"
    )

    return df


class HDFS_LSTMAutoencoder:

    def __init__(
            self,
            model,
            event_to_id,
            n_features,
            max_seq_len
    ):

        self.model = model

        self.event_to_id = event_to_id

        self.n_features = n_features

        self.max_seq_len = max_seq_len

    def prepare_data(self, sequences_list):

        padded = pad_sequences(
            sequences_list,
            maxlen=self.max_seq_len,
            padding='post'
        )

        X = np.zeros(

            (
                len(padded),
                self.max_seq_len,
                self.n_features
            ),

            dtype=np.float32
        )

        for i, seq in enumerate(padded):

            for j, event in enumerate(seq):

                if (
                    event != 0 and
                    event in self.event_to_id
                ):

                    X[
                        i,
                        j,
                        self.event_to_id[event]
                    ] = 1.0

        return X

    def get_errors(
            self,
            sequences_list,
            batch_size=4096
    ):

        errors = []

        total = len(sequences_list)

        for i in range(
                0,
                total,
                batch_size
        ):

            batch = sequences_list[
                    i:i + batch_size
                    ]

            X_batch = self.prepare_data(batch)

            X_pred = self.model.predict(
                X_batch,
                verbose=0
            )

            batch_errors = np.mean(

                np.square(X_batch - X_pred),

                axis=(1, 2)
            )

            errors.extend(batch_errors)

            print(
                f"Обработано: "
                f"{min(i + batch_size, total):,} "
                f"/ {total:,}"
            )

        return np.array(errors)


if __name__ == "__main__":

    LOG_FILE = "HDFS.log"
    LABEL_FILE = "anomaly_label.csv"


    df_blocks = parse_logs(LOG_FILE)

    labels = pd.read_csv(LABEL_FILE)

    if 'BlockId' in labels.columns:

        labels = labels.rename(
            columns={'BlockId': 'block_id'}
        )

    if 'Label' in labels.columns:

        labels = labels.rename(
            columns={'Label': 'label'}
        )

    df = df_blocks.merge(
        labels,
        on='block_id',
        how='inner'
    )

    df['label'] = df['label'].map({

        'Normal': 0,

        'Anomaly': 1

    }).fillna(0)

    print("\nРаспределение классов:")

    print(df['label'].value_counts())


    print("\n" + "=" * 70)
    print("ЗАГРУЗКА ISOLATION FOREST")
    print("=" * 70)

    start_time = time.time()

    mem_before = get_memory_usage_mb()

    iforest, vectorizer, scaler = joblib.load(
        "iforest_model.pkl"
    )

    load_time_iforest = (
            time.time() - start_time
    )

    mem_after = get_memory_usage_mb()

    iforest_memory = (
            mem_after - mem_before
    )

    print("Isolation Forest загружен")


    analysis_start = time.time()

    X_all = vectorizer.transform(
        df['template']
    )

    X_all_scaled = scaler.transform(
        X_all
    )

    if_scores = -iforest.decision_function(
        X_all_scaled
    )

    if_pred = (
        iforest.predict(X_all_scaled) == -1
    ).astype(int)

    analysis_time_iforest = (
            time.time() - analysis_start
    )

    iforest_speed = (
            len(df) / analysis_time_iforest
    )


    print("\n" + "=" * 70)
    print("ЗАГРУЗКА LSTM AUTOENCODER")
    print("=" * 70)

    start_time = time.time()

    mem_before = get_memory_usage_mb()

    lstm_model = load_model(
        "lstm_autoencoder.keras"
    )

    metadata = joblib.load(
        "lstm_metadata.pkl"
    )

    threshold = metadata['threshold']

    lstm_ae = HDFS_LSTMAutoencoder(

        model=lstm_model,

        event_to_id=metadata['event_to_id'],

        n_features=metadata['n_features'],

        max_seq_len=metadata['max_seq_len']
    )

    load_time_lstm = (
            time.time() - start_time
    )

    mem_after = get_memory_usage_mb()

    lstm_memory = (
            mem_after - mem_before
    )

    print("LSTM Autoencoder загружен")


    analysis_start = time.time()

    errors = lstm_ae.get_errors(
        df['event_ids'].tolist()
    )

    lstm_pred = (
        errors > threshold
    ).astype(int)

    analysis_time_lstm = (
            time.time() - analysis_start
    )

    lstm_speed = (
            len(df) / analysis_time_lstm
    )


    print("\n" + "=" * 70)
    print("ENSEMBLE МЕТОД")
    print("=" * 70)

    if_scores_norm = (

        (if_scores - np.min(if_scores)) /

        (np.max(if_scores) - np.min(if_scores))
    )

    errors_norm = (

        (errors - np.min(errors)) /

        (np.max(errors) - np.min(errors))
    )

    ensemble_scores = (

        0.5 * if_scores_norm +

        0.5 * errors_norm
    )

    ensemble_threshold = np.percentile(
        ensemble_scores,
        95
    )

    ensemble_pred = (
        ensemble_scores > ensemble_threshold
    ).astype(int)


    comparison = pd.DataFrame({

        'Model': [

            'Isolation Forest',

            'LSTM Autoencoder',

            'Ensemble'
        ],

        'F1 Score': [

            f1_score(df['label'], if_pred),

            f1_score(df['label'], lstm_pred),

            f1_score(df['label'], ensemble_pred)
        ],

        'ROC AUC': [

            roc_auc_score(df['label'], if_scores),

            roc_auc_score(df['label'], errors),

            roc_auc_score(
                df['label'],
                ensemble_scores
            )
        ],

        'Model Load Time (sec)': [

            load_time_iforest,

            load_time_lstm,

            load_time_iforest + load_time_lstm
        ],

        'Analysis Speed (logs/sec)': [

            iforest_speed,

            lstm_speed,

            (
                    iforest_speed +
                    lstm_speed
            ) / 2
        ],

        'Memory Usage (MB)': [

            iforest_memory,

            lstm_memory,

            iforest_memory + lstm_memory
        ]
    })

    print("\n")
    print("=" * 90)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 90)

    print(comparison.round(4))


    print("\n" + "=" * 70)
    print("ENSEMBLE CLASSIFICATION REPORT")
    print("=" * 70)

    print(

        classification_report(
            df['label'],
            ensemble_pred,
            digits=4
        )
    )


    plt.figure(figsize=(7, 6))

    sns.heatmap(

        confusion_matrix(
            df['label'],
            ensemble_pred
        ),

        annot=True,

        fmt='d',

        cmap='Blues'
    )

    plt.title(
        'Confusion Matrix - Ensemble'
    )

    plt.xlabel('Predicted')

    plt.ylabel('True')

    plt.tight_layout()

    plt.show()


    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    axes[0, 0].bar(
        comparison['Model'],
        comparison['F1 Score']
    )

    axes[0, 0].set_title('F1 Score')

    axes[0, 1].bar(
        comparison['Model'],
        comparison['ROC AUC']
    )

    axes[0, 1].set_title('ROC AUC')

    axes[1, 0].bar(
        comparison['Model'],
        comparison['Model Load Time (sec)']
    )

    axes[1, 0].set_title('Model Load Time')

    axes[1, 1].bar(
        comparison['Model'],
        comparison['Memory Usage (MB)']
    )

    axes[1, 1].set_title('Memory Usage')

    plt.tight_layout()

    plt.show()


    comparison.to_csv(
        "ensemble_model_comparison.csv",
        index=False
    )

    print("\nАнализ завершён.")
    print("Результаты сохранены.")