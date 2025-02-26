import pandas as pd
import numpy as np
import tensorflow as tf

class DataProcessor:
    def __init__(self, true_path: str, fake_path: str):
        """
        Classe para processar e limpar dados de notícias verdadeiras e falsas.

        :param true_path: Caminho do arquivo CSV contendo notícias verdadeiras
        :param fake_path: Caminho do arquivo CSV contendo notícias falsas
        """
        self.true_path = true_path
        self.fake_path = fake_path

    def merge_data(self):
        """
        Processa os dados combinando as colunas 'title' e 'text', atribuindo rótulos
        e embaralhando os dados para futura análise ou modelagem.

        :return: DataFrame processado com colunas 'text', 'status' e 'subject'
        """
        df_true = pd.read_csv(self.true_path)
        df_fake = pd.read_csv(self.fake_path)

        df_true['text'] = df_true['title'] + ' ' + df_true['text']
        df_fake['text'] = df_fake['title'] + ' ' + df_fake['text']
        df_true['status'] = 1
        df_fake['status'] = 0

        df = pd.concat([df_true[['text', 'status', 'subject']], df_fake[['text', 'status', 'subject']]], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)

        return df

    def clean_text(self, text):
        """
        Realiza a padronização do texto, removendo caracteres especiais, números e espaços extras.

        :param text: Texto a ser limpo
        :return: Texto processado
        """
        text = text.lower()
        text = tf.strings.regex_replace(text, '[^\w\s]', '')
        text = tf.strings.regex_replace(text, '\d+', '')
        text = tf.strings.strip(text)
        return text.numpy().decode('utf-8')

    def normalize_and_save(self, output_path: str):
        """
        Processa e limpa os dados, gerando um novo dataset processado.

        :param output_path: Caminho para salvar o novo dataset
        """
        df = self.merge_data()
        df['text'] = df['text'].apply(lambda x: self.clean_text(x))
        df.to_csv(output_path, index=False)
        print(f"Dataset salvo em: {output_path}")

if __name__ == "__main__":
    data_normalize = DataProcessor("Dataset/True.csv", "Dataset/Fake.csv")
    data_normalize.normalize_and_save("Dataset/Processed.csv")
