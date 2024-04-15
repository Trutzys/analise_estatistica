import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import scipy as sp
import scipy.stats
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from google.colab import drive
drive.mount('/content/gdrive')

from google.colab import files
uploaded = files.upload()

sdgIndex = pd.read_csv('/content/gdrive/MyDrive/sdg_index_2000-2022.csv')
sdgIndex.drop(['country_code'], axis = 1, inplace = True)

NOTA = 'sdg_index_score'
NOTA1 = 'goal_1_score'
NOTA2 = 'goal_2_score'
NOTA3 = 'goal_3_score'
NOTA4 = 'goal_4_score'
NOTA5 = 'goal_5_score'
NOTA6 = 'goal_6_score'
NOTA7 = 'goal_7_score'
NOTA8 = 'goal_8_score'
NOTA9 = 'goal_9_score'
NOTA10 = 'goal_10_score'
NOTA11 = 'goal_11_score'
NOTA12 = 'goal_12_score'
NOTA13 = 'goal_13_score'
NOTA14 = 'goal_14_score'
NOTA15 = 'goal_15_score'
NOTA16 = 'goal_16_score'
NOTA17 = 'goal_17_score'

MEDIA = 'Media Geral'
NOPOBR = 'Zero Pobreza'
NOFOME = 'Zero Fome'
SAUDE = 'Saude e Bem-estar'
EDUCA = 'Qualidade da Educação'
IGUAL = 'Igualdade de Genero'
AGUA = 'Agua Limpa e Saneamento'
ENERG = 'Energia Limpa'
ECONO = 'Crecimento Economico'
INFRA = 'Infraestrutura'
DESIG = 'Redução da Desigualdade'
CIDAD = 'Comunidades Sustentaveis'
CONSU = 'Consumo Responsavel'
CLIMA = 'Ação Climatica'
VIDSUB = 'Vida embaixo da água'
VIDTER = 'Vida na Terra'
JUSTI = 'Paz e Justiça'
PARCE = 'Pacerias'


brasilIndex = sdgIndex[sdgIndex['country'] == 'Brazil']
brasilIndex.head(23)


alemanhaIndex = sdgIndex[sdgIndex['country'] == 'Germany']
alemanhaIndex.head(10)


indiaIndex = sdgIndex[sdgIndex['country'] == 'India']
indiaIndex.head(10)


ethiopiaIndex = sdgIndex[sdgIndex['country'] == 'Ethiopia']
ethiopiaIndex.head(10)


score = brasilIndex[NOTA6]
year = brasilIndex['year']

score2 = alemanhaIndex[NOTA6]
year2 = alemanhaIndex['year']

score3 = indiaIndex[NOTA6]
year3 = indiaIndex['year']

score4 = ethiopiaIndex[NOTA6]
year4 = ethiopiaIndex['year']

plt.ylim([0, 100])

plt.plot(year, score, label = 'Brasil')

plt.plot(year2, score2, label = 'Alemanha')

plt.plot(year3, score3, label = 'India')

plt.plot(year4, score4, label = 'Etiopia')

plt.grid()
plt.legend(title = 'Pais')
plt.xlabel('Anos')
plt.ylabel('Nota Atribuida (SDG)')
plt.xlim(2000, 2023)
plt.title('Nota de Acesso a Água Limpa e Saneamento')


score5 = brasilIndex[NOTA7]
year5 = brasilIndex['year']

score6 = alemanhaIndex[NOTA7]
year6 = alemanhaIndex['year']

score7 = indiaIndex[NOTA7]
year7 = indiaIndex['year']

score8 = ethiopiaIndex[NOTA7]
year8 = ethiopiaIndex['year']

plt.ylim([0, 100])

plt.plot(year5, score5, label = 'Brasil')

plt.plot(year6, score6, label = 'Alemanha')

plt.plot(year7, score7, label = 'India')

plt.plot(year8, score8, label = 'Etiopia')

plt.grid()
plt.legend(title = 'Pais')
plt.xlabel('Anos')
plt.ylabel('Nota Atribuida (SDG)')
plt.xlim(2000, 2023)
plt.title('Nota de Custo de Energia e Energia Limpa')


brasilNormal = sdgIndex[sdgIndex['country'] == 'Brazil']
brasilNormal.drop(['country', 'year'], axis = 1, inplace = True)
brasilNormal = brasilNormal.rename(columns = {NOTA : MEDIA, NOTA1 : NOPOBR, NOTA2 : NOFOME, NOTA3 : SAUDE, NOTA4 : EDUCA, NOTA5 : IGUAL, NOTA6 : AGUA, NOTA7 : ENERG, NOTA8 : ECONO, NOTA9 : INFRA, NOTA10 : DESIG,
NOTA11 : CIDAD, NOTA12 : CONSU, NOTA13 : CLIMA, NOTA14 : VIDSUB, NOTA15 : VIDTER, NOTA16 : JUSTI, NOTA17 : PARCE})
sns.heatmap(brasilNormal, yticklabels = False, linewidth = .6, cmap = "coolwarm", vmin = 30, vmax = 100)
plt.xlabel('Categoria')
plt.ylabel('Anos (Decrescente)')
plt.title('Mapa de calor das notas de cada categoria - Brasil')

indiaNormal = sdgIndex[sdgIndex['country'] == 'India']
indiaNormal.drop(['country', 'year'], axis = 1, inplace  = True)
indiaNormal = indiaNormal.rename(columns = {NOTA : MEDIA, NOTA1 : NOPOBR, NOTA2 : NOFOME, NOTA3 : SAUDE, NOTA4 : EDUCA, NOTA5 : IGUAL, NOTA6 : AGUA, NOTA7 : ENERG, NOTA8 : ECONO, NOTA9 : INFRA, NOTA10 : DESIG,
NOTA11 : CIDAD, NOTA12 : CONSU, NOTA13 : CLIMA, NOTA14 : VIDSUB, NOTA15 : VIDTER, NOTA16 : JUSTI, NOTA17 : PARCE})
sns.heatmap(indiaNormal, yticklabels = False, linewidth = .6, cmap = "coolwarm", vmin = 30, vmax = 100)
plt.xlabel('Categoria')
plt.ylabel('Anos (Decrescente)')
plt.title('Mapa de calor das notas de cada categoria - India')


yearS = brasilIndex['year']
scoreS = brasilIndex[NOTA9]
plt.figure(figsize = (10, 5))
plt.grid()
plt.xlabel('Anos')
plt.ylabel('Nota Atribuida (SDG)')
plt.ylim(30, 100)
plt.xlim(2000, 2023)
plt.title('Nota do Index de Infrestrutura - Brasil')
plt.scatter(yearS, scoreS)

nota1 = sdgIndex[NOTA1]
nota3 = sdgIndex[NOTA3]
plt.figure(figsize = (10, 5))
plt.grid()
plt.xlabel('Nota de Zero Pobreza')
plt.ylabel('Nota de Saúde e Bem-Estar')
plt.title('Zero pobreza X Saúde e Bem Estar')
plt.xlim(10, 100)
plt.ylim(0, 100)
plt.scatter(nota1, nota3)


pyplot.hist(brasilIndex[NOTA16])
plt.xlim(54.5, 59.5)
plt.grid()
plt.title('Histograma da Nota de Justiça e Paz (Brasil)')
plt.xlabel('Nota Atribuida')
pyplot.show()

pyplot.hist(brasilIndex[NOTA5])
plt.xlim(62.5, 70.8)
plt.grid()
plt.title('Histograma da Nota da Igualdade de Gênero')
plt.xlabel('Nota Atribuida')
pyplot.show()


size = 100
testx = np.arange(size)
testy = brasilIndex['goal_12_score'].values
testh = plt.hist(testy, density=True)

dist_names = ['expon', 'logistic', 'norm']
for dist_name in dist_names:
  dist = getattr(scipy.stats, dist_name)
  params = dist.fit(testy)
  arg = params[:-2]
  loc = params[-2]
  scale = params[-1]
  if arg:
    pdf_fitted = dist.pdf(testx, *arg, loc=loc, scale=scale)
  else:
    pdf_fitted = dist.pdf(testx, loc=loc, scale=scale)
  plt.plot(pdf_fitted, label = dist_name)

plt.legend(loc = 'upper right')
plt.grid()
plt.title('Consumo Responsável e Produção')
plt.xlabel('Nota')
plt.xlim(83.25, 85.75)
plt.ylim(0, 1)

size = 100
testx = np.arange(size)
testy = brasilIndex['goal_7_score'].values
testh = plt.hist(testy, density=True)

dist_names = ['expon', 'logistic', 'norm']
for dist_name in dist_names:
  dist = getattr(scipy.stats, dist_name)
  params = dist.fit(testy)
  arg = params[:-2]
  loc = params[-2]
  scale = params[-1]
  if arg:
    pdf_fitted = dist.pdf(testx, *arg, loc=loc, scale=scale)
  else:
    pdf_fitted = dist.pdf(testx, loc=loc, scale=scale)
  plt.plot(pdf_fitted, label = dist_name)

plt.legend(loc = 'upper right')
plt.grid()
plt.title('Custo de Energia e Energia Limpa')
plt.xlabel('Nota')
plt.xlim(85, 91)
plt.ylim(0, 1)


brasilIndex[[NOTA8, NOTA3]].cov()

brasilIndex[[NOTA2, NOTA3]].cov()


brasilIndex[[NOTA8, NOTA3]].corr()

brasilIndex[[NOTA2, NOTA3]].corr()


x = brasilNormal.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
tabelaNormal = pd.DataFrame(x_scaled)
tabelaNormal.head(23)

#Visualização da variavel de Infraestrutura normalizada
tabelaNormal[9].head(23)

#Visualização da variavel de Igualdade de Gênero normalizada
tabelaNormal[5].head(23)


brasilIndex[NOTA].value_counts(sort = False).sort_index()

alemanhaIndex[NOTA].value_counts(sort = False).sort_index()

indiaIndex[NOTA].value_counts(sort = False).sort_index()

ethiopiaIndex[NOTA].value_counts(sort = False).sort_index()

mediaBR = np.mean(brasilIndex[NOTA])
print(mediaBR)
desvioBR = np.std(brasilIndex[NOTA])
intervaloBR = scipy.stats.norm.interval(0.10, loc = mediaBR, scale = desvioBR)
print('Intervalo de Confiança da Nota Geral do Brasil é: ')
print(intervaloBR)

mediaAL = np.mean(alemanhaIndex[NOTA])
print(mediaAL)
desvioAL = np.std(alemanhaIndex[NOTA])
intervaloAL = scipy.stats.norm.interval(0.10, loc = mediaAL, scale = desvioAL)
print('Intervalo de Confiança da Nota Geral da Alemanha é: ')
print(intervaloAL)

mediaID = np.mean(indiaIndex[NOTA])
print(mediaID)
desvioID = np.std(indiaIndex[NOTA])
intervaloID = scipy.stats.norm.interval(0.10, loc = mediaID, scale = desvioID)
print('Intervalo de Confiança da Nota Geral da India é: ')
print(intervaloID)

mediaET = np.mean(ethiopiaIndex[NOTA])
print(mediaET)
desvioET = np.std(ethiopiaIndex[NOTA])
intervaloET = scipy.stats.norm.interval(0.10, loc = mediaET, scale = desvioET)
print('Intervalo de Confiança da Nota Geral da Etiopia é: ')
print(intervaloET)
