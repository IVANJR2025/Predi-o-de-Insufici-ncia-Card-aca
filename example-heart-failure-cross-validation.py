import math
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm, metrics, tree
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt

# Read the data file (Heart Failure clinical records Dataset), taken from https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records
df = pd.read_csv("./heart_failure_clinical_records_dataset.csv")

# Print the data for verification
print(df)

# Generate a report for Pandas Dataframe analysis using the ydata profiling library
profile = ProfileReport(df, title="Pandas Profiling Report")
try:
    profile.to_notebook_iframe()
except Exception:
    pass  # Only works in Jupyter Notebook
profile.to_file("report.html")
print("Report saved to report.html - open in browser to view.")

# Export report to PDF
def save_report_pdf():
    """Try weasyprint first, fallback to fpdf2 summary."""
    try:
        from weasyprint import HTML
        HTML("report.html").write_pdf("report.pdf")
        print("Report saved to report.pdf")
        return
    except ImportError:
        pass
    except Exception:
        pass
    # Fallback: create summary PDF with fpdf2
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=14)
        pdf.cell(0, 10, "Heart Failure - Relatorio do Dataset", ln=True, align="C")
        pdf.set_font("Helvetica", size=10)
        pdf.ln(5)
        w = 190  # page width minus margins
        pdf.multi_cell(w, 6, "Relatorio completo: report.html")
        pdf.multi_cell(w, 6, f"Dataset: {len(df)} amostras e {len(df.columns)} colunas.")
        pdf.multi_cell(w, 6, "Features: age, ejection_fraction, serum_creatinine. Target: DEATH_EVENT.")
        pdf.multi_cell(w, 6, "Para PDF completo, abra report.html no navegador e imprima como PDF.")
        pdf.output("report.pdf")
        print("Report summary saved to report.pdf (use report.html for full report)")
    except ImportError:
        print("Install weasyprint or fpdf2 for PDF export: pip install weasyprint fpdf2")

save_report_pdf()

# Reduce the data to the relevant columns (for what we want to do)
print("\nLet's reduce the data to the relevant columns (after analysis and selection).")
df = df[['age', 'ejection_fraction', 'serum_creatinine', 'DEATH_EVENT']]
print(df)

# Put the data in variable X and the answers in variable y
print("\nThe training features are given by the columns: age, ejection_fraction, serum_creatinine")
print("The answers are given by the column: DEATH_EVENT")
#X = np.array(df.drop(['DEATH_EVENT'], 1))
X = np.array(df.drop(columns='DEATH_EVENT'))
print("X_train:\n\n")
print(X)
y = np.array(df['DEATH_EVENT'])
print("\ny_train:\n\n")
print(y)

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
print(X)

# Split the dataset into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the ML model
print("\nNow that we have the data prepared, let's train a model and check its performance")
#clf = LogisticRegression()
#clf = svm.SVC(probability=True)
#clf = tree.DecisionTreeClassifier()
clf = MLPClassifier(random_state=1,max_iter=500)
#clf.fit(X_train, y_train)
#tree.plot_tree(clf) 

#implement the Cross-Validation (CV) technique
print("\nImplementation of the cross-validation technique K-folds.")
cv = KFold(n_splits=5)
print(f"\nNumber of folds: {cv.get_n_splits(X_train)}\n")
print(cv)

scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro")
print("\nModel performance on training set (Cross-Validation):\n")
print(f"\nScores obtained for each of the {cv.get_n_splits(X_train)} validation sets (folds): {scores}")
print(f"\nAverage performance of the {cv.get_n_splits(X_train)} folds: f1-score ({scores.mean():.4f})\tstandard deviation ({scores.std():.4f})")

# Train the final model on the full training set
print("\nNow we train the final model on the full training set and evaluate on the test set.")
clf.fit(X_train, y_train)

# Evaluate on test set
y_pred = clf.predict(X_test)
print("\n=== Evaluation on Test Set ===\n")
print("Classification Report:")
print(metrics.classification_report(y_test, y_pred, target_names=['Survived', 'Death']))
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score (macro): {metrics.f1_score(y_test, y_pred, average='macro'):.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))

# Predict probabilities (when available)
if hasattr(clf, 'predict_proba'):
    y_proba = clf.predict_proba(X_test)
    print("\nPrediction probabilities (first 5 samples):")
    for i in range(min(5, len(y_proba))):
        print(f"  Sample {i+1}: P(survive)={y_proba[i][0]:.3f}, P(death)={y_proba[i][1]:.3f}")

print("\n=== Training complete! ===")

