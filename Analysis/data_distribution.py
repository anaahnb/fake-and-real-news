import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Settings.keys import ParamsKeys

class DataDistribution:
    def __init__(self, true_path: str, fake_path: str, processed_path: str):
        """
        Classe para análise exploratória de um dataset de notícias.

        :param true_path: Caminho do arquivo CSV contendo as notícias verdadeiras.
        :param fake_path: Caminho do arquivo CSV contendo as notícias falsas.
        :param processed_path: Caminho do arquivo CSV contendo os dados processados.
        """

        self.df_true = pd.read_csv(true_path)
        self.df_fake = pd.read_csv(fake_path)

        self.df_processed = pd.read_csv(processed_path)

    def plot_category_distribution(self):
        """
        Plota gráficos de barras mostrando a distribuição das categorias (subject)
        separadamente para notícias falsas e verdadeiras.
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
        colors = sns.color_palette('pastel')

        datasets = [(self.df_fake, "Distribuição das Categorias - Notícias Falsas"),
                    (self.df_true, "Distribuição das Categorias - Notícias Verdadeiras")]

        for i, (df, title) in enumerate(datasets):
            ax = df[ParamsKeys.SUBJECT].value_counts().plot(kind='bar', color=colors, edgecolor=ParamsKeys.EDGE_COLOR, ax=axes[i])
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Categoria", fontsize=12)
            ax.set_ylabel("Contagem", fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    def plot_status_distribution(self):
        """
        Plota um gráfico de pizza mostrando a distribuição do status (0 = Fake, 1 = Real)
        para verificar o balanceamento da base de dados.
        """
        plt.figure(figsize=(7, 7))
        colors = sns.color_palette(ParamsKeys.GRAPHS_THEME)
        labels = [ParamsKeys.FAKE, ParamsKeys.REAL]
        sizes = self.df_processed[ParamsKeys.STATUS].value_counts().values

        wedges, texts, autotexts = plt.pie(sizes, colors=colors,
            autopct='%1.1f%%', startangle=140,
            explode=(0.1, 0), wedgeprops={'edgecolor': ParamsKeys.EDGE_COLOR})

        plt.title("Distribuição do Status das Notícias", fontsize=16)
        plt.legend(wedges, labels, loc="best", fontsize=12)
        plt.axis('equal')
        plt.show()

if __name__ == "__main__":
    analysis = DataDistribution(ParamsKeys.TRUE_DATASET_PATH, ParamsKeys.FAKE_DATASET_PATH, ParamsKeys.NORMALIZED_DATASET_PATH)
    analysis.plot_category_distribution()
    analysis.plot_status_distribution()
