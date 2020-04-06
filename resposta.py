import pandas as pd
import numpy as np
import sklearn
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("train -dados.csv")
df_answer = pd.read_csv("test_TRATADO.csv")

troca_provas_cn = {
    "16f84b7b3d2aeaff7d2f01297e6b3d0e25c77bb2" : 0,
    "b9b06ce8c319a3df2158ea3d0aef0f7d3eecaed7" : 1,
    "2d22ac1d42e6187f09ee6c578df187a760123ccf" : 2,
    "c8328ebc6f3238e06076c481bc1b82b8301e7a3f" : 3,
    "66b1dad288e13be0992bae01e81f71eca1c6e8a6" : 4,
    "03b1fba5c1ebbc47988cd303b08982cfb2aa9cf2" : 5,
    "c87a85497686b3e7b3765f84a2ca95256f0f66aa" : 6,
    "69ed2ddcb151cfebe3d2ae372055335ac7c8c144" : 7,
    "1bcdece8fb1b952552b319e4e5512bbcf540e338" : 8,
    "a27a1efea095c8a973496f0b57a24ac6775d95b0" : 9
}

troca_provas_ch = {
    "9cd70f1b922e02bd33453b3f607f5a644fb9b1b8" : 0,
    "909237ab0d84688e10c0470e2997348aff585273" : 1,
    "2d22ac1d42e6187f09ee6c578df187a760123ccf" : 2,
    "f48d390ab6a2428e659c37fb8a9d00afde621889" : 3,
    "942ab3dc020af4cf53740b6b07e9dd7060b24164" : 4,
    "f94e97c2a5689edd5369740fde9a927e23a9465f" : 5,
    "0fb4772fc6ee9b951ade2fbe6699cc37985c422e" : 6,
    "c95541bf218d7ff70572ca4bcb421edeff05c6d5" : 7,
    "6c3fec2ef505409a9e7c3d2e8634fa2aced4ee93" : 8,
    "d5f6d17523d2cce3e4dc0a7f0582a85cec1c15ee" : 9
}

troca_provas_lc = {
    "01af53cd161a420fff1767129c10de560cc264dd" : 0,
    "2d22ac1d42e6187f09ee6c578df187a760123ccf" : 1,
    "01abbb7f1a90505385f44eec9905f82ca2a42cfd" : 2,
    "5aebe5cad7fabc1545ac7fba07a4e6177f98483c" : 3,
    "72f80e4b3150c627c7ffc93cfe0fa13a9989b610" : 4,
    "9cbf6bf31d9d89a64ce2737ece4834fde4a95029" : 5,
    "fa86b01f07636b15adfd66b688c79934730721a6" : 6,
    "44b09b311799bd684b3d02463bfa99e472c6adb3" : 7,
    "481058938110a64a272266e3892102b8ef0ca96f" : 8
}

troca_provas_mt = {
    "97caab1e1533dba217deb7ef41490f52e459ab01" : 0,
    "2d22ac1d42e6187f09ee6c578df187a760123ccf" : 1,
    "81d0ee00ef42a7c23eb04496458c03d4c5b9c31a" : 2,
    "767a32545304ed293242d528f54d4edb1369f910" : 3,
    "577f8968d95046f5eb5cc158608e12fa9ba34c85" : 4,
    "0ec1c8ac02d2747b6e9a99933fbf96127dd6e89e" : 5,
    "0e0082361eaceb6418bb17305a2b7912650b4783" : 6,
    "6d6961694e839531aec2d35bbd8552b55394a0d7" : 7,
    "73c5c86eef8f70263e4c5708d153cca123f93378" : 8
}

troca_sexo = {
    "M" : 0,
    "F" : 1
}


df['PROVA_CN'] = df.CO_PROVA_CN.map(troca_provas_cn)
df['PROVA_CH'] = df.CO_PROVA_CH.map(troca_provas_ch)
df['PROVA_LC'] = df.CO_PROVA_LC.map(troca_provas_lc)
df['PROVA_MT'] = df.CO_PROVA_MT.map(troca_provas_mt)
df['SEXO']     = df.TP_SEXO.map(troca_sexo)

df_answer['PROVA_CN'] = df_answer.CO_PROVA_CN.map(troca_provas_cn)
df_answer['PROVA_CH'] = df_answer.CO_PROVA_CH.map(troca_provas_ch)
df_answer['PROVA_LC'] = df_answer.CO_PROVA_LC.map(troca_provas_lc)
df_answer['PROVA_MT'] = df_answer.CO_PROVA_MT.map(troca_provas_mt)
df_answer['SEXO']     = df_answer.TP_SEXO.map(troca_sexo)

df = df.drop("CO_PROVA_CN",axis = 1)
df = df.drop("CO_PROVA_CH",axis = 1)
df = df.drop("CO_PROVA_LC",axis = 1)
df = df.drop("CO_PROVA_MT",axis = 1)
df = df.drop("TP_SEXO",axis = 1)
df = df.fillna(0)

df_answer = df_answer.drop("CO_PROVA_CN",axis = 1)
df_answer = df_answer.drop("CO_PROVA_CH",axis = 1)
df_answer = df_answer.drop("CO_PROVA_LC",axis = 1)
df_answer = df_answer.drop("CO_PROVA_MT",axis = 1)
df_answer = df_answer.drop("TP_SEXO",axis = 1)
df_answer = df_answer.fillna(0)


#x = df.drop("NU_NOTA_MT", axis = 1)
x = df[["PROVA_MT","NU_IDADE","CO_UF_RESIDENCIA","NU_NOTA_CN","NU_NOTA_CH","NU_NOTA_LC","TP_COR_RACA","SEXO","TP_ESCOLA","TP_NACIONALIDADE"]]
x_answer = df_answer[["PROVA_MT","NU_IDADE","CO_UF_RESIDENCIA","NU_NOTA_CN","NU_NOTA_CH","NU_NOTA_LC","TP_COR_RACA","SEXO","TP_ESCOLA","TP_NACIONALIDADE"]]
y = df["NU_NOTA_MT"]



treino_x = x[:-4531]
teste_x = x[-4531:]
treino_y = y[:-4531]
teste_y = y[-4531:]

print(treino_x.shape)
print(teste_x.shape)
print(treino_y.shape)
print(teste_y.shape)

lm = LinearRegression()
lm.fit(treino_x,treino_y)

previsoes = lm.predict(treino_x)

linha_exemplo = df.loc[df['NU_INSCR_CONVERT'] == 29]
linha_exemplo = linha_exemplo[["NU_INSCR_CONVERT","PROVA_MT","NU_IDADE","CO_UF_RESIDENCIA","TP_PRESENCA_MT","NU_NOTA_CN","NU_NOTA_CH","NU_NOTA_LC","TP_COR_RACA","SEXO","TP_ESCOLA","TP_NACIONALIDADE"]]
resposta = lm.predict(x_answer)

prediction = pd.DataFrame(resposta,columns=['predictions'])
prediction = prediction.round(2) 
prediction.to_csv('answer.csv')
