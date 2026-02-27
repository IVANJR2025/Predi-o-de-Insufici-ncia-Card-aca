# Predição de Insuficiência Cardíaca (Heart Failure Prediction)

Projeto de Aprendizagem Automática para predição de eventos de morte em pacientes com insuficiência cardíaca, utilizando validação cruzada K-Fold.

## Descrição

Este projeto utiliza o dataset **Heart Failure Clinical Records** (UCI Machine Learning Repository) para treinar modelos de classificação que preveem o evento de morte (`DEATH_EVENT`) com base em características clínicas dos pacientes.

**Fonte dos dados:** [UCI - Heart failure clinical records](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)

## Estrutura do Projeto

```
Example-Predicting-Heart-Failure/
├── example-heart-failure-cross-validation.py   # Script principal
├── heart_failure_clinical_records_dataset.csv # Dataset
├── requirements.txt                           # Dependências Python
├── README.md                                  # Este ficheiro
├── DOCUMENTACAO_PASSO_A_PASSO.md             # Documentação detalhada
├── report.html                                # Relatório de análise (gerado)
└── report.pdf                                 # Relatório em PDF (gerado)
```

## Requisitos

- Python 3.8+
- Dependências listadas em `requirements.txt`

## Instalação

```bash
pip install -r requirements.txt
```

## Execução

```bash
python example-heart-failure-cross-validation.py
```

## Tecnologias Utilizadas

- **pandas** – Manipulação de dados
- **numpy** – Operações numéricas
- **scikit-learn** – Machine Learning (MLPClassifier, validação cruzada, métricas)
- **ydata-profiling** – Análise exploratória e relatórios
- **matplotlib** – Visualizações (importado)
- **weasyprint** – Exportação do relatório HTML para PDF (no Windows pode falhar; usa-se fpdf2 como alternativa)
- **fpdf2** – Geração de PDF resumido quando weasyprint não está disponível

## Características do Modelo

- **Features:** idade (`age`), fração de ejeção (`ejection_fraction`), creatinina sérica (`serum_creatinine`)
- **Target:** `DEATH_EVENT` (0 = sobrevivência, 1 = óbito)
- **Modelo:** MLPClassifier (rede neural)
- **Validação:** K-Fold (5 folds) com métrica F1-score macro
- **Split:** 80% treino, 20% teste

## Outputs Gerados

| Ficheiro      | Descrição                                                                 |
|---------------|---------------------------------------------------------------------------|
| `report.html` | Relatório completo de perfil do dataset (interativo; abrir no navegador) |
| `report.pdf`  | PDF: versão completa (weasyprint) ou resumo (fpdf2) conforme o ambiente   |

## Resultados Esperados

- **Acurácia:** ~70%
- **F1-Score (macro):** ~0.66
- **Cross-validation:** F1-score médio ~0.69 nos 5 folds

## Documentação Completa

Consulte o ficheiro **DOCUMENTACAO_PASSO_A_PASSO.md** para uma descrição detalhada de todos os passos realizados no projeto.

## Licença

Projeto académico – Universidade de Aveiro, Aprendizagem Automática Aplicada.
