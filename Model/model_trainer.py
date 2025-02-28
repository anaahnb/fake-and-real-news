import tensorflow as tf
from tensorflow.keras import layers, losses
from DataEngineering.preprocessing import DataPreprocessing
from Settings.keys import ParamsKeys

class ModelTrainer:
    def __init__(self, vocab_size, max_length, embedding_dim=16, num_classes=2):
        """
        Classe responsável pela criação, compilação e treinamento do modelo de detecção de notícias falsas.

        :param vocab_size: Tamanho do vocabulário usado na tokenização.
        :param max_length: Comprimento máximo das sequências de entrada.
        :param embedding_dim: Dimensão do vetor de embedding.
        :param num_classes: Número de classes para a classificação (padrão: 2).
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """
        Cria e retorna o modelo de rede neural.

        :return: Modelo Keras compilado.
        """
        model = tf.keras.Sequential([
            layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.max_length),
            layers.Dropout(0.2),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='sigmoid')
        ])

        model.compile(
            loss=losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer='adam',
            metrics=['accuracy']
        )

        return model

    def train(self, train_dataset, val_dataset, epochs=10, patience=3):
        """
        Treina o modelo com os datasets fornecidos.

        :param train_dataset: Dataset de treino formatado como `tf.data.Dataset`.
        :param val_dataset: Dataset de validação.
        :param epochs: Número máximo de épocas para o treinamento.
        :param patience: Número de épocas sem melhora antes da interrupção antecipada.
        :return: Histórico de treinamento.
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[early_stopping]
        )

        return history

    def evaluate(self, test_dataset):
        """
        Avalia o modelo no dataset de teste.

        :param test_dataset: Dataset de teste formatado como `tf.data.Dataset`.
        :return: Acurácia do modelo no conjunto de teste.
        """
        loss, acc = self.model.evaluate(test_dataset)
        print(f"Test Accuracy: {acc * 100:.2f}%")
        return acc

if __name__ == "__main__":
    preprocessing = DataPreprocessing(ParamsKeys.TRUE_DATASET_PATH, ParamsKeys.FAKE_DATASET_PATH)
    train_dataset, val_dataset, test_dataset, vocab_size, max_length = preprocessing.process_and_split()

    model = ModelTrainer(vocab_size, max_length)
    model.train(train_dataset, val_dataset)
    model.evaluate(test_dataset)
