import pandas as pd
import os
import re

class DataNormalize:
    def __init__(self, true_path: str, fake_path: str):
        """
        Classe para processar e limpar dados de notícias verdadeiras e falsas.

        :param true_path: Caminho do arquivo CSV contendo notícias verdadeiras
        :param fake_path: Caminho do arquivo CSV contendo notícias falsas
        """
        self.true_path = true_path
        self.fake_path = fake_path
        self.dataset = None

    def merge_data(self):
        """
        Processa os dados combinando as colunas 'title' e 'text', atribuindo rótulos,
        embaralhando os dados e padronizando o texto.

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

        df['text'] = df['text'].apply(self.clean_text)

        self.dataset = df
        return df

    def clean_text(self, text):
        """
        Realiza a padronização do texto, removendo caracteres especiais, números e espaços extras.

        :param text: Texto a ser limpo
        :return: Texto processado
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.strip()
        return text

    def save_processed_data(self, output_path="Dataset/processed/Nomalized.csv"):
        """
        Salva o DataFrame processado em um arquivo CSV, garantindo que o diretório exista.

        :param output_path: Caminho para salvar o arquivo processado
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if self.dataset is not None:
            self.dataset.to_csv(output_path, index=False)
            print(f"Arquivo salvo como: {output_path}")
        else:
            print("Erro: O dataset ainda não foi processado. Execute merge_data() primeiro.")

if __name__ == "__main__":
    normalizer = DataNormalize("Dataset/True.csv", "Dataset/Fake.csv")
    df_processed = normalizer.merge_data()
    normalizer.save_processed_data()
