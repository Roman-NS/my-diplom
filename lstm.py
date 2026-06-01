import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import joblib

from collections import defaultdict

warnings.filterwarnings('ignore')

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    RepeatVector,
    TimeDistributed
)

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    classification_report
)


MODEL_DIR = "saved_models"

os.makedirs(MODEL_DIR, exist_ok=True)


def parse_logs_optimized(log_file="HDFS.log"):

    config = TemplateMinerConfig()
    config.sim_th = 0.5
    config.depth = 4

    parser = TemplateMiner(config=config)

    block_data = defaultdict(
        lambda: {
            'event_ids': [],
            'raw_count': 0
        }
    )

    line_count = 0

    print("Парсинг данных...")

    with open(
        log_file,
        'r',
        encoding='utf-8',
        errors='ignore'
    ) as f:

        for line in f:

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

            block_data[block_id]['event_ids'].append(
                result["cluster_id"]
            )

            block_data[block_id]['raw_count'] += 1

            line_count += 1

            if line_count % 2_000_000 == 0:

                print(
                    f"Обработано строк: "
                    f"{line_count:,}"
                )

    print(
        f"\nПарсинг завершён. "
        f"Блоков: {len(block_data):,}"
    )

    data = []

    for block_id, info in block_data.items():

        data.append({

            'block_id': block_id,

            'event_ids': info['event_ids'],

            'event_count': info['raw_count']
        })

    df = pd.DataFrame(data)

    print(
        f"DataFrame создан. "
        f"Размер: {df.shape}"
    )

    return df


class HDFS_LSTMAutoencoder:

    def __init__(self, max_seq_len=40):

        self.max_seq_len = max_seq_len

        self.model = None

        self.event_to_id = None

        self.n_features = None


    def fit_vocabulary(self, all_sequences):

        all_events = np.unique(
            np.concatenate(all_sequences)
        )

        self.event_to_id = {

            event: idx + 1

            for idx, event
            in enumerate(all_events)
        }

        self.n_features = (
            len(self.event_to_id) + 1
        )

        print(
            f"Словарь событий создан.\n"
            f"Уникальных событий: "
            f"{self.n_features}"
        )


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


    def build(self):

        self.model = Sequential([

            LSTM(
                64,
                activation='relu',
                input_shape=(
                    self.max_seq_len,
                    self.n_features
                ),
                return_sequences=True
            ),

            LSTM(
                32,
                activation='relu',
                return_sequences=False
            ),

            RepeatVector(
                self.max_seq_len
            ),

            LSTM(
                32,
                activation='relu',
                return_sequences=True
            ),

            LSTM(
                64,
                activation='relu',
                return_sequences=True
            ),

            TimeDistributed(
                Dense(self.n_features)
            )
        ])

        self.model.compile(
            optimizer='adam',
            loss='mse'
        )

        print(
            "Модель LSTM Autoencoder построена"
        )


    def train(
            self,
            X_train,
            epochs=8,
            batch_size=64
    ):

        print(
            f"Обучение на "
            f"{X_train.shape[0]} "
            f"примерах..."
        )

        self.model.fit(
            X_train,
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )


    def get_reconstruction_errors_batch(
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
                verbose=0,
                batch_size=128
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

            del X_batch, X_pred

        return np.array(errors)


def plot_reconstruction_error_distribution(
        errors,
        labels,
        threshold
):

    plt.figure(figsize=(12, 8))

    normal_errors = errors[labels == 0]

    anomaly_errors = errors[labels == 1]

    sns.histplot(
        normal_errors,
        bins=100,
        kde=True,
        color='blue',
        label='Нормальные блоки',
        alpha=0.7,
        stat="density"
    )

    sns.histplot(
        anomaly_errors,
        bins=100,
        kde=True,
        color='red',
        label='Аномальные блоки',
        alpha=0.7,
        stat="density"
    )

    plt.axvline(
        threshold,
        color='black',
        linestyle='--',
        linewidth=2.5,
        label='Порог аномальности'
    )

    plt.title(
        'Распределение ошибок реконструкции',
        fontsize=14,
        fontweight='bold'
    )

    plt.xlabel(
        'Reconstruction Error',
        fontsize=12
    )

    plt.ylabel(
        'Density',
        fontsize=12
    )

    plt.legend(fontsize=11)

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(

        'lstm_reconstruction_error_distribution.png',

        dpi=300,

        bbox_inches='tight'
    )

    print(
        "\nГрафик сохранён:\n"
        "lstm_reconstruction_error_distribution.png"
    )

    plt.show(block=True)


def save_model(
        lstm_ae,
        threshold
):

    model_path = os.path.join(
        MODEL_DIR,
        "lstm_autoencoder.keras"
    )

    lstm_ae.model.save(model_path)

    metadata = {

        'event_to_id': lstm_ae.event_to_id,

        'n_features': lstm_ae.n_features,

        'max_seq_len': lstm_ae.max_seq_len,

        'threshold': threshold
    }

    metadata_path = os.path.join(
        MODEL_DIR,
        "lstm_metadata.pkl"
    )

    joblib.dump(
        metadata,
        metadata_path
    )

    print("\n" + "=" * 70)

    print("МОДЕЛЬ СОХРАНЕНА")

    print("=" * 70)

    print(f"\nФайлы модели:")

    print(f" - {model_path}")

    print(f" - {metadata_path}")


if __name__ == "__main__":

    df_blocks = parse_logs_optimized(
        "HDFS.log"
    )

    labels_df = pd.read_csv(
        "anomaly_label.csv"
    )

    if 'BlockId' in labels_df.columns:

        labels_df = labels_df.rename(

            columns={
                'BlockId': 'block_id'
            }
        )

    if 'Label' in labels_df.columns:

        labels_df = labels_df.rename(

            columns={
                'Label': 'label'
            }
        )

    df = df_blocks.merge(
        labels_df,
        on='block_id',
        how='inner'
    )

    df['label'] = df['label'].map({

        'Normal': 0,

        'Anomaly': 1

    }).fillna(0)

    print(
        "\nРаспределение классов:\n",
        df['label'].value_counts()
    )


    lstm_ae = HDFS_LSTMAutoencoder(
        max_seq_len=40
    )

    lstm_ae.fit_vocabulary(
        df['event_ids'].tolist()
    )

    normal_sample = df[
        df['label'] == 0
        ].sample(
        n=100000,
        random_state=42
    )

    X_train = lstm_ae.prepare_data(
        normal_sample['event_ids'].tolist()
    )

    lstm_ae.build()

    lstm_ae.train(
        X_train,
        epochs=8,
        batch_size=64
    )


    print(
        "\nВычисление ошибок реконструкции..."
    )

    errors = lstm_ae.get_reconstruction_errors_batch(
        df['event_ids'].tolist(),
        batch_size=4096
    )

    threshold = np.percentile(
        errors,
        95
    )

    lstm_pred = (
            errors > threshold
    ).astype(int)


    print(
        "\nПостроение графика..."
    )

    plot_reconstruction_error_distribution(
        errors,
        df['label'].values,
        threshold
    )


    print("\n" + "=" * 85)

    print("РЕЗУЛЬТАТЫ LSTM AUTOENCODER")

    print("=" * 85)

    print(
        classification_report(
            df['label'],
            lstm_pred,
            digits=4
        )
    )

    print(
        f"F1-score (Anomaly) = "
        f"{f1_score(df['label'], lstm_pred):.4f}"
    )

    print(
        f"ROC AUC = "
        f"{roc_auc_score(df['label'], errors):.4f}"
    )

    print(
        f"Порог аномальности = "
        f"{threshold:.6f}"
    )


    save_model(
        lstm_ae,
        threshold
    )

    print(
        "\nПрограмма завершена."
    )