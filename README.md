## 🌟 Visão Geral
**Objetivo:** Classificar clientes em "baixo risco" (0) ou "alto risco" (1) de inadimplência com base em características como histórico de crédito, idade, emprego, etc.

**Técnicas Utilizadas:**
- Pré-processamento: Codificação de variáveis categóricas (`pd.get_dummies`), padronização.
- Modelagem: Regressão Logística, Random Forest, XGBoost.
- Métricas: Acurácia, Precisão, Recall, F1-Score, AUC-ROC.

- Principais Bibliotecas:

pandas, numpy (análise de dados)

scikit-learn, xgboost (modelos)

matplotlib, seaborn (visualização)

streamlit (deploy)

- 📊 Resultados
Desempenho dos Modelos
Modelo	Acurácia	F1-Score (Classe 1)	AUC-ROC
Regressão Logística	77%	0.58	0.76
Random Forest	78%	0.52	0.74
XGBoost	80%	0.63	0.81
Variáveis Mais Importantes (Random Forest):

credit_amount (Valor do crédito)

duration (Duração do empréstimo)

age (Idade do cliente)
