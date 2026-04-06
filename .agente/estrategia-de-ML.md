Estratégia de Machine Learning: Modelo de Regressão

Este documento descreve como transformar os 10.000 processos históricos numa inteligência preditiva capaz de estimar o tempo de sentença em dias.

1. Porquê o XGBoost?

Para dados estruturados (tabelas), o XGBoost (Extreme Gradient Boosting) é o estado da arte.

Vantagem: Lida bem com dados faltantes e variáveis categóricas.

Precisão: Capta padrões complexos (ex: comportamento específico de um juiz baseado no valor da causa).

2. Preparação do Dataset (O "Vetor" de Treino)

Variáveis de Entrada (Features - X):

Comarca: Label Encoding (Ex: Boa Vista = 1).

Assunto: Categoria jurídica.

Valor da Causa: Numérico normalizado.

Juiz: ID do juiz.

Presença de Advogado: Booleano.

Similaridade Média (RAG): Média de dias dos 5 processos mais parecidos via busca vetorial.

Alvo da Predição (Target - y):

Dias Totais: Tempo decorrido entre petição inicial e sentença.

3. Pipeline de Processamento (Exemplo Python)

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Carregar os 10.000 processos
df = pd.read_sql("SELECT * FROM processos_historicos", db_connection)

# 2. Engenharia de Features (Label Encoding)
df['comarca_cat'] = df['comarca'].astype('category').cat.codes
df['juiz_cat'] = df['juiz'].astype('category').cat.codes

# 3. Divisão do Dataset (80% treino, 20% teste)
X = df[['comarca_cat', 'juiz_cat', 'valor_causa', 'assunto_cat', 'similaridade_rag']]
y = df['dias_totais']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Treino do Modelo
model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)


4. Integração Híbrida (RAG + ML)

O segredo é a combinação:

Busca Vetorial (RAG): Encontra casos "espelho" para confiança visual do usuário.

Modelo XGBoost: Usa os dados dos casos similares + estatísticas do tribunal para o cálculo final.

5. Como Medir o Sucesso?

Utilizaremos o MAE (Mean Absolute Error).

Se o MAE for 15, o modelo erra por cerca de 15 dias.

Objetivo: Erro inferior a 30 dias é considerado excelente para o setor jurídico.

6. Evolução: Otimização por Comarca

Treinamento de micro-modelos específicos (ex: Juizado Especial Cível vs. Varas de Família) para captar nuances processuais distintas.