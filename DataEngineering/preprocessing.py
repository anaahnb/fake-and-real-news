import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from collections import Counter
from DataEngineering.normalize import DataNormalize
from Settings.keys import ParamsKeys

class DataPreprocessing:
    def __init__(self, true_path: str, fake_path: str):
        """
        Classe responsável pelo pré-processamento dos dados e criação dos datasets para treinamento.

        :param true_path: Caminho do CSV contendo notícias verdadeiras
        :param fake_path: Caminho do CSV contendo notícias falsas
        """
        self.processor = DataNormalize(true_path, fake_path)
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.tokenizer = Tokenizer()

    def tokenize_and_pad(self, X_train, X_val, X_test):
        """
        Tokeniza os textos e aplica padding para garantir que todas as sequências tenham o mesmo comprimento.

        :param X_train: Lista de textos do conjunto de treino.
        :param X_val: Lista de textos do conjunto de validação.
        :param X_test: Lista de textos do conjunto de teste.
        :return: Sequências tokenizadas e preenchidas (train, val, test), vocab_size e max_length.
        """
        self.tokenizer.fit_on_texts(X_train)

        train_seq = self.tokenizer.texts_to_sequences(X_train)
        val_seq = self.tokenizer.texts_to_sequences(X_val)
        test_seq = self.tokenizer.texts_to_sequences(X_test)

        vocab_size = len(self.tokenizer.word_index) + 1
        max_length = max(len(sequence) for sequence in train_seq)

        train_seq = pad_sequences(train_seq, maxlen=max_length, padding='post', truncating='post')
        val_seq = pad_sequences(val_seq, maxlen=max_length, padding='post', truncating='post')
        test_seq = pad_sequences(test_seq, maxlen=max_length, padding='post', truncating='post')

        return train_seq, val_seq, test_seq, vocab_size, max_length

    def process_and_split(self):
        """
        Normaliza os dados, divide em treino, validação e teste e retorna datasets TensorFlow.
        """
        df = self.processor.merge_data()
        df[ParamsKeys.TEXT] = df[ParamsKeys.TEXT].apply(self.processor.clean_text)

        y = df[ParamsKeys.STATUS].values
        X = df[ParamsKeys.TEXT].values
        del df

        X_raw, X_test, y_raw, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)
        del X_raw, y_raw

        print("Distribuição das classes:")
        print("Treino:", Counter(y_train))
        print("Teste:", Counter(y_test))
        print("Validação:", Counter(y_val))

        train_seq, val_seq, test_seq, vocab_size, max_length = self.tokenize_and_pad(X_train, X_val, X_test)

        def create_tf_dataset(X, y):
            """
            Converte arrays NumPy em um `tf.data.Dataset` otimizado para treinamento.

            :param X: Sequências tokenizadas.
            :param y: Rótulos correspondentes.
            :return: `tf.data.Dataset` pronto para uso.
            """
            dataset = tf.data.Dataset.from_tensor_slices((X, y))
            return dataset.cache().shuffle(len(X)).batch(32).prefetch(buffer_size=self.AUTOTUNE)

        train_dataset = create_tf_dataset(train_seq, y_train)
        val_dataset = create_tf_dataset(val_seq, y_val)
        test_dataset = create_tf_dataset(test_seq, y_test)

        return train_dataset, val_dataset, test_dataset, vocab_size, max_length, y_train, y_test, y_val

if __name__ == "__main__":
    preprocessing = DataPreprocessing(ParamsKeys.TRUE_DATASET_PATH, ParamsKeys.FAKE_DATASET_PATH)
    train_dataset, val_dataset, test_dataset, vocab_size, max_length = preprocessing.process_and_split()
