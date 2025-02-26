import pandas as pd
import matplotlib.pyplot as plt

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
        self.dataset['subject'].value_counts().plot(kind='bar', color='skyblue')
        plt.title("Distribuição de Categorias das Notícias")
        plt.xlabel("Categoria")
        plt.ylabel("Contagem")
        plt.xticks(rotation=45)
        plt.show()

    def plot_status_distribution(self):
        """
        Plota um gráfico de barras mostrando a distribuição do status (0 = Fake, 1 = Real)
        para verificar o balanceamento da base de dados.
        """
        plt.figure(figsize=(5, 5))
        self.dataset['status'].value_counts().plot(kind='bar', color=['red', 'green'])
        plt.title("Distribuição do Status das Notícias")
        plt.xlabel("Status")
        plt.ylabel("Contagem")
        plt.xticks(ticks=[0, 1], labels=["Fake", "Real"], rotation=0)
        plt.show()

if __name__ == "__main__":
    analysis = DataAnalysis("Dataset/Processed.csv")
    analysis.plot_category_distribution()
    analysis.plot_status_distribution()
