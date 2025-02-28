import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from Settings.keys import ParamsKeys

class DataSizeAnalysis:
    def __init__(self, true_path: str, fake_path: str, processed_path: str):
        """
        Classe para análise comparativa do tamanho das notícias antes e depois do processamento.

        :param true_path: Caminho do arquivo CSV contendo as notícias verdadeiras brutas.
        :param fake_path: Caminho do arquivo CSV contendo as notícias falsas brutas.
        :param processed_path: Caminho do arquivo CSV contendo os dados processados.
        """
        self.df_true_raw = pd.read_csv(true_path)
        self.df_fake_raw = pd.read_csv(fake_path)

        self.df_processed = pd.read_csv(processed_path)

    def calculate_avg_characters(self):
        """
        Calcula e exibe a média do número de caracteres por notícia para cada categoria
        em formato de tabela estilizada usando tabulate.
        """
        datasets = {
            "Notícias Verdadeiras - Brutas": self.df_true_raw[ParamsKeys.TEXT],
            "Notícias Falsas - Brutas": self.df_fake_raw[ParamsKeys.TEXT],
            "Notícias Verdadeiras - Processadas": self.df_processed[self.df_processed[ParamsKeys.STATUS] == 1][ParamsKeys.TEXT],
            "Notícias Falsas - Processadas": self.df_processed[self.df_processed[ParamsKeys.STATUS] == 0][ParamsKeys.TEXT]
        }

        data_summary = []

        for name, text_data in datasets.items():
            text_data = text_data.dropna()
            avg_chars = text_data.apply(len).mean()
            data_summary.append([name, f"{avg_chars:.2f} caracteres"])

        # Criar tabela estilizada
        table = tabulate(data_summary, headers=["Categoria", "Média de Caracteres"], tablefmt="fancy_grid")

        print(table)


    def plot_length_distribution(self):
        """
        Plota histogramas comparando o tamanho das notícias antes e depois do processamento,
        com gráficos separados para cada conjunto de dados.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)

        datasets = [
            (self.df_true_raw[ParamsKeys.TEXT], "Notícias Verdadeiras - Brutas", "blue", axes[0, 0]),
            (self.df_fake_raw[ParamsKeys.TEXT], "Notícias Falsas - Brutas", "red", axes[0, 1]),
            (self.df_processed[self.df_processed[ParamsKeys.STATUS] == 1][ParamsKeys.TEXT], "Notícias Verdadeiras - Processadas", "purple", axes[1, 0]),
            (self.df_processed[self.df_processed[ParamsKeys.STATUS] == 0][ParamsKeys.TEXT], "Notícias Falsas - Processadas", "orange", axes[1, 1])
        ]

        for text_data, title, color, ax in datasets:
            text_data = text_data.dropna()
            text_lengths = [len(news.split()) for news in text_data if len(news.split()) > 0]

            if text_lengths:
                text_lengths = np.log(text_lengths)
                sns.histplot(text_lengths, bins=20, color=color, kde=True, ax=ax)
                ax.set_title(title, fontsize=14)
                ax.set_xlabel("Log do Comprimento da Notícia")
                ax.set_ylabel("Densidade")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    analysis = DataSizeAnalysis(ParamsKeys.TRUE_DATASET_PATH, ParamsKeys.FAKE_DATASET_PATH, ParamsKeys.NORMALIZED_DATASET_PATH)
    analysis.calculate_avg_characters()
    analysis.plot_length_distribution()
