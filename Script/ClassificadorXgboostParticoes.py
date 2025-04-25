# Importa bibliotecas essenciais para manipulação de dados, processamento de texto e aprendizado de máquina.
import pandas as pd  # Para manipulação de dados estruturados.
from sklearn.feature_extraction.text import TfidfVectorizer  # Para converter texto em vetores TF-IDF.
from sklearn.model_selection import train_test_split, GridSearchCV  # Para divisão de dados e busca de hiperparâmetros.
from sklearn.metrics import classification_report  # Para gerar relatórios de classificação.
from xgboost import XGBClassifier  # Importa o classificador XGBoost.
from nltk.corpus import stopwords  # Importa stopwords do NLTK
import nltk

nltk.download('stopwords')  # Garante que as stopwords estão disponíveis

def vectorize_text(X_train, X_test):
    # Obtém as stopwords em português do NLTK
    portuguese_stopwords = stopwords.words('portuguese')
    # Cria um vetorizador TF-IDF com as stopwords em português
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=portuguese_stopwords)
    X_train_tfidf = vectorizer.fit_transform(X_train)  # Ajusta o vetorizador e transforma os dados de treino
    X_test_tfidf = vectorizer.transform(X_test)  # Transforma os dados de teste com o mesmo vetorizador
    return X_train_tfidf, X_test_tfidf, vectorizer


def train_xgb_model(X_train_tfidf, y_train):
    xgb_model = XGBClassifier(eta=0.3, gamma=1, eval_metric='logloss')  # Inicializa o classificador XGBoost.
    xgb_model.fit(X_train_tfidf, y_train)  # Treina o modelo XGBoost com os dados processados.
    return xgb_model

def evaluate_model(model, X_test_tfidf, y_test):
    y_pred = model.predict(X_test_tfidf)  # Faz previsões com o modelo.
    print(classification_report(y_test, y_pred))  # Gera um relatório de classificação detalhado.

def perform_grid_search(X_train_tfidf, y_train):
    param_grid = {
         'eta': [0, 0.3, 1],  # Taxa de aprendizado
        'gamma': [0.1, 1, 10] # Parâmetro de regularização.
    }
    xgb_model = XGBClassifier(eval_metric='logloss')  # Cria o modelo base.
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='accuracy',  # Métrica de avaliação.
        cv=10,  # Número de folds para validação cruzada.
        verbose=2,  # Nível de detalhamento do log.
        n_jobs=-1  # Utiliza todos os núcleos disponíveis.
    )
    grid_search.fit(X_train_tfidf, y_train)  # Realiza o Grid Search nos dados de treino.
    print("Melhores parâmetros:", grid_search.best_params_)  # Exibe os melhores hiperparâmetros encontrados.
    print("Melhor accuracy:", grid_search.best_score_)  # Exibe a melhor pontuação obtida.
    return grid_search.best_estimator_


def main():
    train_data = pd.read_csv('Data/train.csv')
    test_data = pd.read_csv('Data/test.csv')

    # Divide os dados em texto e rótulos
    X_train = train_data['text']
    y_train = train_data['label']
    X_test = test_data['text']
    y_test = test_data['label']
    
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
    xgb_model = train_xgb_model(X_train_tfidf, y_train)
    evaluate_model(xgb_model, X_test_tfidf, y_test)
    best_model = perform_grid_search(X_train_tfidf, y_train)
    evaluate_model(best_model, X_test_tfidf, y_test)

if __name__ == "__main__":
    main()
