# Experimentos de Detecção de Discurso de Ódio Utilizando Somente o XGBoost

## Descrição Geral
Este repositório documenta um experimento de classificação binária utilizando o classificador **XGBoost** para a detecção de discurso de ódio em textos em português. Este experimento foi baseado no experimento orginal https://github.com/Carlosbera7/ExperimentoBaseOriginal utilizando as partições já disponibilizadas em https://github.com/Carlosbera7/SalvarParticoes

## Objetivo
O objetivo principal é avaliar a eficácia do modelo **XGBoost** em classificar textos como discurso de ódio ou não, nãu utilizando LSTM como proposto no experimento original.

## Metodologia

### Etapas do Experimento
1. **Carga e Pré-processamento dos Dados**:
   - Os dados foram divididos em conjuntos de treino e teste (90% para treino e 10% para teste) com estratificação para manter a proporção das classes.

2. **Vetorização com TF-IDF**:
   - Foi utilizado o **TfidfVectorizer** para converter os textos em vetores numéricos, eliminando stopwords em português.

3. **Treinamento Inicial**:
   - O classificador **XGBoost** foi treinado com parâmetros padrão para criar um modelo inicial.

4. **Busca de Hiperparâmetros**:
   - Utilizou-se a técnica de Grid Search para encontrar os melhores valores de `eta` (taxa de aprendizado) e `gamma` (regularização).

5. **Avaliação do Modelo**:
   - Foram gerados dados de classificação com métricas como precision, recall e F1-score.

### Ferramentas e Bibliotecas
- **Pandas**: Manipulação e análise de dados estruturados.
- **Scikit-learn**: Vetorização TF-IDF, divisão de dados, e Grid Search.
- **XGBoost**: Algoritmo principal para classificação.
- **NLTK**: Tratamento de stopwords em português.

### Parâmetros Ajustados no Grid Search
- **eta** (taxa de aprendizado): `[0, 0.3, 1]`
- **gamma** (regularização): `[0.1, 1, 10]`

## Estrutura do Código
- Transforma textos em vetores TF-IDF.
- Treina o modelo XGBoost com hiperparâmetros padrão.
- Avalia o modelo gerando um relatório de classificação.
- Realiza busca de hiperparâmetros com validação cruzada.

## Resultados
Os resultados incluem:
- **Relatórios de métricas**: Precision, recall, f1-score e accuracy.
- **Melhores parâmetros do XGBoost** obtidos via grid search.

Exemplo de saída:
```

Melhores parâmetros: {'eta': 0.3, 'gamma': 1.0}
Melhor f1-score: 0.872182
              precision    recall  f1-score   support

           0       0.87      0.97      0.92       431
           1       0.85      0.54      0.66       136

    accuracy                           0.87       567
   macro avg       0.86      0.76      0.79       567
weighted avg       0.87      0.87      0.86       567
```
![XgboostParticoes](https://github.com/user-attachments/assets/47e8c088-bae3-40fa-ab03-b5f2790a9718)

## Estrutura do Repositório
- [`Scripts/ClassificadorOriginalParticoes.py`](https://github.com/Carlosbera7/ExperimentoOriginalParticoesXgboost/blob/main/Script/ClassificadorXgboostParticoes.py): Script principal para executar o experimento.
- [`Data/`](https://github.com/Carlosbera7/ExperimentoOriginalParticoesXgboost/tree/main/Data): Pasta contendo o conjunto de dados e o Embeddings GloVe pré-treinados (necessário para execução).
- [`Execução`](https://fluffy-adventure-p4xvpvwx5vc7q56.github.dev/): O código pode ser executado diretamente no ambiente virtual.

