import os
import kaggle
from Settings.keys import ParamsKeys

class DataLoading:
    def __init__(self, dataset_name: str, download_path: str = ParamsKeys.DATASET_FOLDER_PATH):
        """
        Classe para baixar datasets do Kaggle.

        :param dataset_name: Nome do dataset no formato "autor/dataset"
        :param download_path: Caminho onde o dataset será salvo
        """
        self.dataset_name = dataset_name
        self.download_path = download_path

        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

    def download(self, unzip: bool = True):
        """
        Baixa o dataset do Kaggle e descompacta se necessário.

        :param unzip: Se True, descompacta os arquivos baixados
        """
        try:
            kaggle.api.dataset_download_files(
                self.dataset_name,
                path=self.download_path,
                unzip=unzip
            )
            print(f"Dataset '{self.dataset_name}' baixado com sucesso em '{self.download_path}'.")
        except Exception as e:
            print(f"Erro ao baixar o dataset: {e}")

if __name__ == "__main__":
    downloader = DataLoading(ParamsKeys.DATASET_URL_DOWNLOAD)
    downloader.download()