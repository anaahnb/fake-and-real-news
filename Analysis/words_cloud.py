import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class WordCloudGenerator:
    def __init__(self, dataset_path: str):
        """
        Classe para gerar nuvens de palavras a partir das notícias falsas e verdadeiras.

        :param dataset_path: Caminho do arquivo CSV contendo os dados processados
        """
        self.dataset = pd.read_csv(dataset_path)

    def generate_wordcloud(self, text_data, title: str):
        """
        Gera e exibe uma nuvem de palavras com base no texto fornecido.

        :param text_data: Lista de textos para gerar a nuvem de palavras
        :param title: Título do gráfico gerado
        """
        text = " ".join(text_data)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title)
        plt.axis("off")
        plt.show()

    def generate_fake_and_real_wordclouds(self):
        """
        Gera e exibe nuvens de palavras separadas para notícias falsas e verdadeiras.
        """
        real_news = self.dataset[self.dataset['status'] == 1]['text'].tolist()
        fake_news = self.dataset[self.dataset['status'] == 0]['text'].tolist()

        self.generate_wordcloud(real_news, "Nuvem de Palavras - Notícias Verdadeiras")
        self.generate_wordcloud(fake_news, "Nuvem de Palavras - Notícias Falsas")

if __name__ == "__main__":
    wordcloud_gen = WordCloudGenerator("Dataset/Processed.csv")
    wordcloud_gen.generate_fake_and_real_wordclouds()
