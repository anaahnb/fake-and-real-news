import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataDistribution:
    def __init__(self, dataset_path: str):
        """
        Classe para análise exploratória de um dataset de notícias.

        :param dataset_path: Caminho do arquivo CSV contendo os dados processados
        """
        self.dataset = pd.read_csv(dataset_path)

    def plot_category_distribution(self):
        """
        Plota um gráfico de barras mostrando a distribuição das categorias (subject) das notícias.
        """
        plt.figure(figsize=(10, 5))
        colors = sns.color_palette('pastel')
        ax = self.dataset['subject'].value_counts().plot(kind='bar', color=colors, edgecolor='black')
        plt.title("Distribuição de Categorias das Notícias", fontsize=16)
        plt.xlabel("Categoria", fontsize=12)
        plt.ylabel("Contagem", fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def plot_status_distribution(self):
        """
        Plota um gráfico de pizza mostrando a distribuição do status (0 = Fake, 1 = Real)
        para verificar o balanceamento da base de dados.
        """
        plt.figure(figsize=(7, 7))
        colors = sns.color_palette('pastel')
        labels = ["Fake", "Real"]
        sizes = self.dataset['status'].value_counts().values

        wedges, texts, autotexts = plt.pie(sizes, colors=colors,
            autopct='%1.1f%%', startangle=140,
            explode=(0.1, 0), wedgeprops={'edgecolor': 'black'})

        plt.title("Distribuição do Status das Notícias", fontsize=16)
        plt.legend(wedges, labels, loc="best", fontsize=12)
        plt.axis('equal')
        plt.show()

if __name__ == "__main__":
    analysis = DataDistribution("Dataset/Processed.csv")
    analysis.plot_category_distribution()
    analysis.plot_status_distribution()
