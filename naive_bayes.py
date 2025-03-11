# Manipular os dados
import pandas as pd
# Divide o conjunto de dados em duas partes: uma para treinamento e outra para teste.
from sklearn.model_selection import train_test_split
# implementa o Naive Bayes Gaussiano, um classificador probabilístico baseado 
# no teorema de Bayes, assumindo que as características seguem uma distribuição normal (gaussiana).
from sklearn.naive_bayes import GaussianNB
#Converte rótulos categóricos (como texto) em valores numéricos
# para que possam ser usados em modelos de machine learning.
from sklearn.preprocessing import LabelEncoder
# confusion_matrix: Calcula a matriz de confusão, que mostra o desempenho de um classificador, 
# comparando as previsões com os valores reais.
#Avaliar o desempenho de modelos de classificação (ex.: True Positives, False Positives, etc.).
# acuracu_score: Calcula a acurácia do modelo, ou seja, a porcentagem de previsões corretas.
from sklearn.metrics import confusion_matrix, accuracy_score
# Um visualizador da matriz de confusão, tornando os resultados mais fáceis de interpretar, 
# exibindo-os graficamente.
from yellowbrick.classifier import ConfusionMatrix
# Ler o dataset
dados = pd.read_csv('Credit.csv')
print(dados.head())
print(dados.describe())
print(dados.shape)

# formato da matriz
previsores = dados.iloc[:,0:20].values
classe = dados.iloc[:, 20].values

# Transformação dos atributos categóricos em atributos numéricos, 
# passando o índice de cada coluna categórica. Precisamos criar um objeto para cada atributo categórico, 
# pois na sequência vamos executar o processo de encoding novamente para o registro de teste
# Se forem utilizados objetos diferentes, o número atribuído a cada valor poderá ser diferente,
# o que deixará o teste inconsistente.
# Codificação de variáveis categóricas para variáveis numéricas.
labelencoder1 = LabelEncoder()
previsores[:,0] = labelencoder1.fit_transform(previsores[:,0])

labelencoder2 = LabelEncoder()
previsores[:,2] = labelencoder2.fit_transform(previsores[:,2])

labelencoder3 = LabelEncoder()
previsores[:, 3] = labelencoder3.fit_transform(previsores[:, 3])

labelencoder4 = LabelEncoder()
previsores[:, 5] = labelencoder4.fit_transform(previsores[:, 5])

labelencoder5 = LabelEncoder()
previsores[:, 6] = labelencoder5.fit_transform(previsores[:, 6])

labelencoder6 = LabelEncoder()
previsores[:, 8] = labelencoder6.fit_transform(previsores[:, 8])

labelencoder7 = LabelEncoder()
previsores[:, 9] = labelencoder7.fit_transform(previsores[:, 9])

labelencoder8 = LabelEncoder()
previsores[:, 11] = labelencoder8.fit_transform(previsores[:, 11])

labelencoder9 = LabelEncoder()
previsores[:, 13] = labelencoder9.fit_transform(previsores[:, 13])

labelencoder10 = LabelEncoder()
previsores[:, 14] = labelencoder10.fit_transform(previsores[:, 14])

labelencoder11 = LabelEncoder()
previsores[:, 16] = labelencoder11.fit_transform(previsores[:, 16])

labelencoder12 = LabelEncoder()
previsores[:, 18] = labelencoder12.fit_transform(previsores[:, 18])

labelencoder13 = LabelEncoder()
previsores[:, 19] = labelencoder13.fit_transform(previsores[:, 19])


# Divisão da base de dados entre treinamento e teste (30% para testar e 70% para treinar)
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                  classe,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
print(x_teste)  


# Criação e treinamento do modelo (geração da tabela de probabilidades)
naive_bayes = GaussianNB()
#treina o modelo
naive_bayes.fit(x_treinamento, y_treinamento)

# Previsões utilizando os registros de teste
previsoes = naive_bayes.predict(x_teste)
print(previsoes)


#geração da matriz de confusão
#A matriz de confusão é uma ferramenta essencial para avaliar a performance de um modelo de classificação, 
# pois mostra não apenas os acertos do modelo (TP e TN), mas também os erros (FP e FN). 
# Isso ajuda a entender onde o modelo está errando e pode fornecer informações valiosas para ajustar o
# modelo ou o processo de treinamento
confusao = confusion_matrix(y_teste, previsoes)
print(confusao)

# calcula a taxa de acerto e a taxa de erro do modelo
taxa_acerto = accuracy_score(y_teste, previsoes)
taxa_erro = 1 - taxa_acerto
#Taxa de acerto: 0.71, Taxa de erro:0.29000000000000004 
print(f'Taxa de acerto: {taxa_acerto}\nTaxa de erro:{taxa_erro}')


# Visualização da matriz de confusão
# Warning interno da biblioteca yellowbrick, já esta na última versão (sem solução para o warning no momento)
v = ConfusionMatrix(GaussianNB())
v.fit(x_treinamento, y_treinamento)
v.score(x_teste, y_teste)
v.poof()


# Previsão com novo registro, transformando os atributos categóricos em numéricos
novo_credito = pd.read_csv('NovoCredit.csv')
novo_credito.shape
#novo_credito

# Usamos o mesmo objeto que foi criado antes, para manter o padrão dos dados
# Chamamos somente o método "transform", pois a adaptação aos dados (fit) já foi feita anteriormente
novo_credito = novo_credito.iloc[:,0:20].values
novo_credito[:,0] = labelencoder1.transform(novo_credito[:,0])
novo_credito[:, 2] = labelencoder2.transform(novo_credito[:, 2])
novo_credito[:, 3] = labelencoder3.transform(novo_credito[:, 3])
novo_credito[:, 5] = labelencoder4.transform(novo_credito[:, 5])
novo_credito[:, 6] = labelencoder5.transform(novo_credito[:, 6])
novo_credito[:, 8] = labelencoder6.transform(novo_credito[:, 8])
novo_credito[:, 9] = labelencoder7.transform(novo_credito[:, 9])
novo_credito[:, 11] = labelencoder8.transform(novo_credito[:, 11])
novo_credito[:, 13] = labelencoder9.transform(novo_credito[:, 13])
novo_credito[:, 14] = labelencoder10.transform(novo_credito[:, 14])
novo_credito[:, 16] = labelencoder11.transform(novo_credito[:, 16])
novo_credito[:, 18] = labelencoder12.transform(novo_credito[:, 18])
novo_credito[:, 19] = labelencoder13.transform(novo_credito[:, 19])


# Resultado da previsão
print(naive_bayes.predict(novo_credito))
