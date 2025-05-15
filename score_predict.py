# Case: Score de crédito dos clientes
# Criar um programa de análise de score do cliente
# Bom(Good), Ok(Standard) ou Ruim(Poor)

# Passo 1: Importar a base de dados
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score


tabela = pd.read_csv("clientes.csv")

# Passo 2: Ler e analisar a base de dados para IA
# print(tabela.info())

# LabelEncoder -> atribui ao texto um número

# profissao
codificador_profissao = LabelEncoder()
tabela["profissao"] = codificador_profissao.fit_transform(tabela["profissao"])
# print (codificador_profissao.classes_)

# mix-credito
codificador_credito = LabelEncoder()
tabela["mix_credito"] = codificador_credito.fit_transform(tabela["mix_credito"])
# print (codificador_credito.classes_)

# comportamento_pagamento
codificador_comportamento = LabelEncoder()
tabela["comportamento_pagamento"] = codificador_comportamento.fit_transform(tabela["comportamento_pagamento"])
# print(codificador_comportamento.classes_)

# separar as informações da base de dados para a IA
# separar a base em x e y (relação entre as colunas)
# x -> todas as outras colunas
x = tabela.drop(["score_credito", "id_cliente"], axis = 1)
# y -> quem eu quero prever (coluna score_credito)
y = tabela["score_credito"]

# separar em dados de treino e teste

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y)

# Criar o modelo de IA:
# passo 1: Importar o modelo (dois modelos nesse caso para comparação de qual é melhor)
# Arvore de decisão -> RandomForest
# Vizinhos Próximos -> KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# criar o modelo
modelo_arvore = RandomForestClassifier()
# modelo_kn = KNeighborsClassifier()

# treinar o modelo
modelo_arvore.fit(x_treino, y_treino)
# modelo_kn.fit(x_treino,y_treino)

# previões e acuracia

previsao_arvore = modelo_arvore.predict(x_teste)
# print(accuracy_score(y_teste,previsao_arvore)) = 83%
# previsao_kn = modelo_kn.predict(x_teste)
# print(accuracy_score(y_teste, previsao_kn)) = 74%

tabela_nova = pd.read_csv("novos_clientes.csv")
# profissao

tabela_nova["profissao"] = codificador_profissao.fit_transform(tabela_nova["profissao"])

# mix_credito
tabela_nova["mix_credito"] = codificador_credito.fit_transform(tabela_nova["mix_credito"])

# comportamento
tabela_nova["comportamento_pagamento"] = codificador_comportamento.fit_transform(tabela_nova["comportamento_pagamento"])


previsao = modelo_arvore.predict(tabela_nova)
tabela_nova["score_credito"] = previsao
# print(tabela_nova)
print(accuracy_score(y_teste,previsao_arvore))
print(previsao)
print(tabela_nova["score_credito"])