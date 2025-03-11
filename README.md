# machine learning naive bayes
 
# Análise de Crédito com Naive Bayes

## Descrição do Projeto
Este projeto utiliza um modelo de classificação Naive Bayes Gaussiano para analisar dados de crédito. O objetivo é prever a aprovação ou reprovação de um crédito com base em um conjunto de variáveis fornecidas no dataset.

## Tecnologias Utilizadas
- **Python**
- **Pandas** (Manipulação de dados)
- **Scikit-learn** (Treinamento do modelo e avaliação)
- **Yellowbrick** (Visualização da matriz de confusão)

## Estrutura do Projeto
1. **Carregamento dos Dados**: O dataset `Credit.csv` é carregado para análise.
2. **Pré-processamento**:
   - Separação entre variáveis preditoras e alvo.
   - Transformação de variáveis categóricas em numéricas com `LabelEncoder`.
   - Divisão dos dados entre treinamento (70%) e teste (30%).
3. **Treinamento do Modelo**:
   - Utilização do classificador **Naive Bayes Gaussiano**.
   - Treinamento com os dados processados.
4. **Avaliação do Modelo**:
   - Predição dos resultados do conjunto de teste.
   - Cálculo da matriz de confusão.
   - Cálculo da taxa de acerto e erro.
   - Visualização da matriz de confusão com Yellowbrick.
5. **Predição com Novos Dados**:
   - Processamento e transformação de um novo conjunto de dados (`NovoCredit.csv`).
   - Predição com o modelo treinado.


## Resultados Esperados
- Um modelo treinado que pode prever com alta acurácia a aprovação ou não de um crédito.
- Visualização da matriz de confusão para análise do desempenho do modelo.

## Autor
Desenvolvido por Mateus Gabriel

