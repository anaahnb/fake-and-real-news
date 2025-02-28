import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from Model.model_trainer import ModelTrainer
from DataEngineering.preprocessing import DataPreprocessing
from Settings.keys import ParamsKeys

class NewsEvaluator:
    def __init__(self, model, history):
        """
        Classe responsável pela avaliação do modelo de detecção de notícias falsas.

        :param model: Modelo treinado para avaliação.
        :param history: Histórico de treinamento do modelo.
        """
        self.model = model
        self.history = history

    def evaluate(self, X_train, y_train, X_test, y_test, X_val, y_val):
        """
        Avalia o modelo e exibe métricas de desempenho.

        :param X_train: Dados de treino tokenizados.
        :param y_train: Rótulos dos dados de treino.
        :param X_test: Dados de teste tokenizados.
        :param y_test: Rótulos dos dados de teste.
        :param X_val: Dados de validação tokenizados.
        :param y_val: Rótulos dos dados de validação.
        """
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        y_pred_val = self.model.predict(X_val)

        y_pred_train = np.argmax(y_pred_train, axis=1)
        y_pred_test = np.argmax(y_pred_test, axis=1)
        y_pred_val = np.argmax(y_pred_val, axis=1)

        print(f'Acurácia de treino: {accuracy_score(y_train, y_pred_train) * 100:.2f} %')
        print(f'Acurácia de teste: {accuracy_score(y_test, y_pred_test) * 100:.2f} %')
        print(f'Acurácia de validação: {accuracy_score(y_val, y_pred_val) * 100:.2f} %')


        print(f'Classification Report (Train) : \n\n{classification_report(y_train, y_pred_train)}')
        print('-----------------------------------------------------')
        print(f'\nClassification Report (Test)  : \n\n{classification_report(y_test, y_pred_test)}')
        print('-----------------------------------------------------')
        print(f'\nClassification Report (Validation)  : \n\n{classification_report(y_val, y_pred_val)}')


        self.plot_training_history()

        self.plot_confusion_matrices(y_train, y_pred_train, y_test, y_pred_test, y_val, y_pred_val)

    def plot_training_history(self):
        """ Plota os gráficos de perda e acurácia do treinamento."""
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Perda no treinamento e na validação')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Acurácia')
        plt.title('Acurácia no Treinamento e na Validação')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self, y_train, y_pred_train, y_test, y_pred_test, y_val, y_pred_val):
        """ Plota as matrizes de confusão para os conjuntos de treino e teste.

        :param y_train: Rótulos verdadeiros do conjunto de treino.
        :param y_pred_train: Rótulos previstos pelo modelo no conjunto de treino.
        :param y_test: Rótulos verdadeiros do conjunto de teste.
        :param y_pred_test: Rótulos previstos pelo modelo no conjunto de teste.
        :param y_val: Rótulos verdadeiros do conjunto de validação.
        :param y_pred_val: Rótulos previstos pelo modelo no conjunto de validação.
        """
        train_matrix = confusion_matrix(y_train, y_pred_train)
        test_matrix = confusion_matrix(y_test, y_pred_test)

        class_labels = ['Falso', 'Verdadeiro']

        disp_train = ConfusionMatrixDisplay(confusion_matrix=train_matrix, display_labels=class_labels)
        disp_test = ConfusionMatrixDisplay(confusion_matrix=test_matrix, display_labels=class_labels)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        disp_train.plot(ax=axs[0], cmap='YlGnBu', colorbar=False)
        axs[0].set_title('Train Confusion Matrix')

        disp_test.plot(ax=axs[1], cmap='YlGnBu', colorbar=False)
        axs[1].set_title('Test Confusion Matrix')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    preprocessing = DataPreprocessing(ParamsKeys.TRUE_DATASET_PATH, ParamsKeys.FAKE_DATASET_PATH)
    train_dataset, val_dataset, test_dataset, vocab_size, max_length, y_train, y_test, y_val = preprocessing.process_and_split()

    model_trainer = ModelTrainer(vocab_size, max_length)
    history = model_trainer.train(train_dataset, val_dataset)

    evaluator = NewsEvaluator(model_trainer.model, history)
    evaluator.evaluate(train_dataset, y_train, test_dataset, y_test, val_dataset, y_val)
