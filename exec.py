import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Dados do conjunto
data = {
    'Sexo': ['F', 'F', 'M', 'F', 'M'],
    'Cirurgia?': ['N', 'N', 'S', 'N', 'N'],
    'TSH': [6.8, 7.3, 8.8, 6.9, 5.4],
    'TT4': [156.2, 152.9, 148.4, 132.7, 150.9],
    'TI': ['N', 'S', 'N', 'N', 'N']
}

# Convertendo para DataFrame
df = pd.DataFrame(data)

# Codificação dos atributos qualitativos para quantitativos
df['Sexo'] = df['Sexo'].map({'F': 0, 'M': 1})
df['Cirurgia?'] = df['Cirurgia?'].map({'N': 0, 'S': 1})
df['TI'] = df['TI'].map({'N': 0, 'S': 1})

# Normalização dos valores dos atributos TSH e TT4 entre 0 e 1
scaler = MinMaxScaler()
df[['TSH', 'TT4']] = scaler.fit_transform(df[['TSH', 'TT4']])

# Dados do novo paciente
new_patient = {'Sexo': 0, 'Cirurgia?': 0, 'TSH': 7, 'TT4': 150, 'TI': 0}
new_patient = pd.DataFrame([new_patient])

# Normalizar os atributos do novo paciente usando o mesmo scaler
new_patient[['TSH', 'TT4']] = scaler.transform(new_patient[['TSH', 'TT4']])

# Calculando a distância euclidiana
distances = np.sqrt(((df[['Sexo', 'Cirurgia?', 'TSH', 'TT4', 'TI']] - new_patient.values) ** 2).sum(axis=1))
distances
