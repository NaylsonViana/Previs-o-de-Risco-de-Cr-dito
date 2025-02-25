import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

#PRÉ-PROCESSAMENTO E EDA
# 1. Carregar dados via ucimlrepo
statlog_german_credit_data = fetch_ucirepo(id=144)

# Extrair features e target
X = statlog_german_credit_data.data.features
# Converter o target para 0 (Bom) e 1 (Ruim)
y = statlog_german_credit_data.data.targets['class'].replace({1: 0, 2: 1})

# Extrair metadados das variáveis
variables_df = statlog_german_credit_data.variables

# Criar dicionário de mapeamento (ex: {'Attribute1': 'checking_account_status', ...})
column_mapping = {}
for index, row in variables_df.iterrows():
    original_name = row['name']  # Ex: 'Attribute1', 'Attribute2', etc.
    new_name = (
        row['description']
        .lower()  # Converter para minúsculas
        .replace(' ', '_')  # Substituir espaços por underscores
        .replace('/', '_')
        .replace('(', '')
        .replace(')', '')
        .replace('__', '_')  # Remover underscores duplos
    )
    column_mapping[original_name] = new_name

# Renomear colunas no DataFrame X
X = X.rename(columns=column_mapping)

# Verificar os novos nomes
print("\nColunas renomeadas:")
print(X.columns.tolist())


# 2. Análise Exploratória
print("\nInformações das Variáveis:")
print(statlog_german_credit_data.variables)

print("\nDistribuição do Target:")
print(y.value_counts())

# Plotar distribuição do target
plt.figure(figsize=(6, 4))
# Changed plt.countplot to sns.countplot
#sns.countplot(x=y)  
#plt.title('Distribuição de Risco de Crédito (0 = Bom, 1 = Ruim)')
#plt.show()

print("\nValores únicos do target (y):", np.unique(y))
print("Contagem de classes:\n", pd.Series(y).value_counts())

#Verificar os dados antes do processamento
print("\nValores únicos em 'status_of_existing_checking_account' ANTES do pré-processamento:")
print(X['status_of_existing_checking_account'].unique())

# 3. Pré-processamento
# Colunas categóricas (usar os nomes reais após renomeação)
cat_cols = [
    'status_of_existing_checking_account',  # Descrição: "Status of existing checking account"
    'purpose',  # Descrição: "Purpose of the loan"
    'personal_status_and_sex',  # Descrição: "Personal status and sex"
    'credit_history',  # Descrição: "Credit history"
    'savings_account_bonds',  # Descrição: "Savings account/bonds"
    'present_employment_since',  # Descrição: "Present employment since"
    'other_debtors__guarantors',
    'property',  # Descrição: "Property"
    'other_installment_plans',  # Descrição: "Other installment plans"
    'housing',  # Descrição: "Housing"
    'job',  # Descrição: "Job"
    'telephone',  # Descrição: "Telephone"
    'foreign_worker'  # Descrição: "Foreign worker"
]

# Colunas numéricas (usar nomes reais)
num_cols = [
    'duration',  # Descrição: "Duration in month"
    'credit_amount',  # Descrição: "Credit amount"
    'installment_rate_in_percentage_of_disposable_income',  # Descrição: "Installment rate in percentage of disposable income"
    'age'  # Descrição: "Age in years"
]

# Aplicar LabelEncoder nas categóricas
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col])

# Padronizar numéricas
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# 4. Engenharia de Features (exemplo: criar razão crédito/idade)
# Changed 'credit_amount' to 'credit_amount', 'age' to 'age_in_years'
X['credit_to_age_ratio'] = X['credit_amount'] / X['age']


#Pós-processamento
print("\nValores únicos em 'status_of_existing_checking_account' APÓS pré-processamento:")
print(X['status_of_existing_checking_account'].unique())


#Análise de Correlação
# Adicionar target ao DataFrame para análise
df_analysis = X.copy()
df_analysis['target'] = y

# Selecionar apenas colunas numéricas para correlação
numerical_cols = df_analysis.select_dtypes(include=['number']).columns

# Matriz de correlação apenas para colunas numéricas
plt.figure(figsize=(14, 10))
sns.heatmap(df_analysis[numerical_cols].corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlação entre Variáveis e Target')
plt.show()



#MODELAGEM
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Verificar colunas não numéricas
non_numeric_cols = X.select_dtypes(include=['object']).columns.tolist()
print("Colunas não numéricas:", non_numeric_cols)

# Regressão Logística
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("Regressão Logística:\n", classification_report(y_test, y_pred))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest:\n", classification_report(y_test, y_pred_rf))

#XGBoost
xgb = XGBClassifier()
xgb.fit(X_train, y_train)  # Agora y_train contém 0 e 1
y_pred_xgb = xgb.predict(X_test)
print("XGBoost:\n", classification_report(y_test, y_pred_xgb))


#Feature Importance
# Importâncias das variáveis (Random Forest)
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10))
plt.title('Top 10 Variáveis Mais Importantes (Random Forest)')
plt.show()

#SALVANDO O MODELO
import joblib
joblib.dump(rf, 'best_credit_model.pkl')
joblib.dump(scaler, 'scaler.pkl')