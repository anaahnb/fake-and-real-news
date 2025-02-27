import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from normalize import DataNormalize
from Settings.keys import ParamsKeys


class DataPreprocessing:
    def __init__(self, true_path: str, fake_path: str):
        """
        Classe responsável por pré-processar os dados e dividi-los em conjuntos de treino, validação e teste.

        :param true_path: Caminho do arquivo CSV contendo notícias verdadeiras
        :param fake_path: Caminho do arquivo CSV contendo notícias falsas
        """
        self.processor = DataNormalize(true_path, fake_path)

    def process_and_split(self, output_dir: str):
        """
        Normaliza os dados, divide em treino, validação e teste, e salva os arquivos.

        :param output_dir: Diretório onde os arquivos processados serão salvos.
        """
        df = self.processor.merge_data()
        df[ParamsKeys.TEXT] = df[ParamsKeys.TEXT].apply(self.processor.clean_text)

        y = df[ParamsKeys.STATUS].values
        X = df[ParamsKeys.TEXT].values
        del df

        # Primeira divisão: 80% para treino + validação, 20% para teste
        X_raw, X_test, y_raw, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Segunda divisão: separando treino e validação
        X_train, X_val, y_train, y_val = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)
        del X_raw, y_raw

        print("Distribuição das classes:")
        print("Treino:", Counter(y_train))
        print("Teste:", Counter(y_test))
        print("Validação:", Counter(y_val))

        train = pd.DataFrame(X_train, columns=[ParamsKeys.TEXT])
        train[ParamsKeys.STATUS] = y_train

        test = pd.DataFrame(X_test, columns=[ParamsKeys.TEXT])
        test[ParamsKeys.STATUS] = y_test

        val = pd.DataFrame(X_val, columns=[ParamsKeys.TEXT])
        val[ParamsKeys.STATUS] = y_val

        os.makedirs(output_dir, exist_ok=True)

        train.to_csv(f"{output_dir}/{ParamsKeys.TRAIN_DATASET_PATH}", index=False)
        val.to_csv(f"{output_dir}/{ParamsKeys.VAL_DATASET_PATH}", index=False)
        test.to_csv(f"{output_dir}/{ParamsKeys.TEST_DATASET_PATH}", index=False)

        print(f"Arquivos salvos em {output_dir}")

if __name__ == "__main__":
    preprocessing = DataPreprocessing(ParamsKeys.TRUE_DATASET_PATH, ParamsKeys.FAKE_DATASET_PATH)
    preprocessing.process_and_split("Dataset/processed")
