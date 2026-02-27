# Documentação Passo a Passo – Predição de Insuficiência Cardíaca

**Projeto:** Predição de eventos de morte em pacientes com insuficiência cardíaca  
**Disciplina:** Aprendizagem Automática Aplicada  
**Universidade de Aveiro**

---

## Índice

1. [Introdução e Objetivos](#1-introdução-e-objetivos)
2. [Configuração do Ambiente](#2-configuração-do-ambiente)
3. [Carregamento e Exploração dos Dados](#3-carregamento-e-exploração-dos-dados)
4. [Análise Exploratória (ProfileReport)](#4-análise-exploratória-profilereport)
5. [Pré-processamento e Seleção de Features](#5-pré-processamento-e-seleção-de-features)
6. [Divisão dos Dados (Treino/Teste)](#6-divisão-dos-dados-treino-teste)
7. [Modelo e Validação Cruzada](#7-modelo-e-validação-cruzada)
8. [Treino Final e Avaliação](#8-treino-final-e-avaliação)
9. [Outputs e Relatórios Gerados](#9-outputs-e-relatórios-gerados)
10. [Alterações Realizadas para Execução](#10-alterações-realizadas-para-execução)

---

## 1. Introdução e Objetivos

O projeto utiliza o dataset **Heart Failure Clinical Records** da UCI Machine Learning Repository para:

- Analisar dados clínicos de pacientes com insuficiência cardíaca
- Gerar relatórios de análise exploratória (HTML e PDF)
- Treinar um modelo de classificação binária para prever o evento de morte (`DEATH_EVENT`)
- Validar o desempenho com validação cruzada K-Fold
- Avaliar o modelo no conjunto de teste

**Dataset:** [Heart failure clinical records](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)

---

## 2. Configuração do Ambiente

### 2.1 Dependências Python

Criou-se o ficheiro `requirements.txt` com as bibliotecas necessárias:

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
ydata-profiling>=4.0.0
matplotlib>=3.5.0
weasyprint>=60.0
```

### 2.2 Instalação

```bash
pip install -r requirements.txt
```

### 2.3 Imports Utilizados

| Biblioteca | Uso |
|------------|-----|
| `pandas` | Leitura e manipulação do CSV |
| `numpy` | Arrays e operações numéricas |
| `sklearn.preprocessing` | StandardScaler para normalização |
| `sklearn.model_selection` | train_test_split, KFold, cross_val_score |
| `sklearn.linear_model` | LogisticRegression (alternativa) |
| `sklearn.svm` | SVC (alternativa) |
| `sklearn.tree` | DecisionTreeClassifier (alternativa) |
| `sklearn.neural_network` | MLPClassifier (modelo utilizado) |
| `sklearn` (metrics) | classification_report, accuracy_score, f1_score, confusion_matrix |
| `ydata_profiling` | ProfileReport para análise exploratória |
| `matplotlib.pyplot` | Visualizações (importado) |
| `weasyprint` | Conversão do relatório HTML para PDF |

---

## 3. Carregamento e Exploração dos Dados

### Passo 3.1

Leitura do ficheiro CSV:

```python
df = pd.read_csv("./heart_failure_clinical_records_dataset.csv")
```

### Passo 3.2

Visualização inicial do dataframe (`print(df)`):

- **299 linhas** (amostras)
- **13 colunas** (variáveis clínicas)

**Colunas do dataset:**
- `age` – idade
- `anaemia` – anemia (0/1)
- `creatinine_phosphokinase` – nível da enzima CPK
- `diabetes` – diabetes (0/1)
- `ejection_fraction` – fração de ejeção (%)
- `high_blood_pressure` – hipertensão (0/1)
- `platelets` – plaquetas
- `serum_creatinine` – creatinina sérica
- `serum_sodium` – sódio sérico
- `sex` – sexo
- `smoking` – fumador (0/1)
- `time` – tempo de acompanhamento
- `DEATH_EVENT` – evento de morte (0/1) — **variável alvo**

---

## 4. Análise Exploratória (ProfileReport)

### Passo 4.1

Geração do relatório com **ydata-profiling**:

```python
profile = ProfileReport(df, title="Pandas Profiling Report")
```

### Passo 4.2

Exportação do relatório:

1. **HTML** – `profile.to_file("report.html")`  
   - Relatório interativo para abrir no navegador.
2. **PDF** – duas opções:
   - **weasyprint** (conversão do HTML para PDF quando disponível, p. ex. em Linux);
   - **fpdf2** (fallback): gera um PDF resumido quando o weasyprint não está disponível (ex.: Windows).
   - Para obter o relatório completo em PDF: abrir `report.html` no navegador e usar "Imprimir" → "Guardar como PDF".

### Conteúdo do relatório

- Estatísticas descritivas de cada variável
- Gráficos de distribuição
- Matriz de correlação
- Deteção de valores em falta e duplicados
- Alertas sobre a qualidade dos dados

---

## 5. Pré-processamento e Seleção de Features

### Passo 5.1 – Seleção de colunas

Redução às variáveis relevantes para o problema:

```python
df = df[['age', 'ejection_fraction', 'serum_creatinine', 'DEATH_EVENT']]
```

- **Features:** `age`, `ejection_fraction`, `serum_creatinine`  
- **Target:** `DEATH_EVENT`

### Passo 5.2 – Separação X e y

```python
X = np.array(df.drop(columns='DEATH_EVENT'))
y = np.array(df['DEATH_EVENT'])
```

### Passo 5.3 – Normalização

Aplicação de `StandardScaler` para média 0 e desvio padrão 1:

```python
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
```

---

## 6. Divisão dos Dados (Treino/Teste)

Divisão 80% treino / 20% teste com `random_state=42`:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- **Treino:** 239 amostras  
- **Teste:** 60 amostras  

---

## 7. Modelo e Validação Cruzada

### Passo 7.1 – Escolha do modelo

Utilização do **MLPClassifier** (rede neural):

```python
clf = MLPClassifier(random_state=1, max_iter=500)
```

**Modelos alternativos (comentados):**
- `LogisticRegression()`
- `svm.SVC(probability=True)`
- `tree.DecisionTreeClassifier()`

### Passo 7.2 – Validação cruzada K-Fold

```python
cv = KFold(n_splits=5)
scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro")
```

- **5 folds**
- **Métrica:** F1-score macro (equilibra precision e recall entre as duas classes)

### Passo 7.3 – Resultados da validação cruzada

- **Scores por fold:** [0.672, 0.748, 0.614, 0.743, 0.682]
- **Média:** 0.6917  
- **Desvio padrão:** 0.0497  

---

## 8. Treino Final e Avaliação

### Passo 8.1 – Treino do modelo

```python
clf.fit(X_train, y_train)
```

### Passo 8.2 – Previsões no conjunto de teste

```python
y_pred = clf.predict(X_test)
```

### Passo 8.3 – Métricas

| Métrica | Valor |
|---------|-------|
| **Acurácia** | 0.70 (70%) |
| **F1-Score (macro)** | 0.6625 |

### Passo 8.4 – Classification Report

```
              precision    recall  f1-score   support
    Survived       0.69      0.89      0.78        35
       Death       0.73      0.44      0.55        25
```

### Passo 8.5 – Matriz de confusão

```
[[31  4]   ← Survived: 31 corretos, 4 falsos positivos
 [14 11]]  ← Death: 11 corretos, 14 falsos negativos
```

### Passo 8.6 – Probabilidades

O MLPClassifier fornece `predict_proba`; são mostradas as probabilidades das primeiras 5 amostras de teste.

---

## 9. Outputs e Relatórios Gerados

| Ficheiro | Descrição |
|----------|-----------|
| `report.html` | Relatório de perfil do dataset (ydata-profiling) |
| `report.pdf` | Versão em PDF do relatório de análise |
| Saída na consola | Dados, métricas, classification report e matriz de confusão |

---

## 10. Alterações Realizadas para Execução

Para permitir execução não interativa do script, foram feitas as seguintes alterações:

### 10.1 – Substituição de `input()` por `print()`

- **Antes:** o script pausava em várias etapas (`input("Press <enter> to continue...")`).
- **Depois:** as mensagens são impressas com `print()` sem interação do utilizador.

### 10.2 – Tratamento de `to_notebook_iframe()`

- **Antes:** `profile.to_notebook_iframe()` falhava fora do Jupyter.
- **Depois:** envolto em `try/except` para evitar erros quando executado como script.

### 10.3 – Exportação para PDF

- Adicionada conversão do relatório HTML para PDF com **weasyprint**.
- Incluído `weasyprint` em `requirements.txt`.

### 10.4 – Ficheiros criados

- **README.md** – descrição do projeto e instruções de uso.
- **DOCUMENTACAO_PASSO_A_PASSO.md** – este documento.
- **requirements.txt** – dependências do projeto.

---

## Notas Finais

- O MLPClassifier pode emitir avisos de convergência; aumentar `max_iter` (ex.: 1000 ou 2000) pode melhorar o ajuste.
- O modelo tem melhor recall para a classe "Survived" (0.89) do que para "Death" (0.44).
- Para utilização interativa em Jupyter, pode-se manter `input()` e `profile.to_notebook_iframe()`.

---

*Documento gerado no âmbito do projeto de Predição de Insuficiência Cardíaca – Universidade de Aveiro.*
