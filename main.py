import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score

# ---------- Data Processing ----------
df = pd.read_csv('loan_data.csv')
df_data = pd.get_dummies(df, columns=['purpose'], drop_first=True)

print(df_data)

# อัตราส่วนค่างวดต่อรายได้จริง 
df_data['installment_to_income'] = df_data['installment'] / np.exp(df_data['log.annual.inc'])

# ผสม fico กับ int.rate — ถ้า fico สูงแต่ดอกเบี้ยก็สูง = ผิดปกติ น่าสงสัย
df_data['fico_rate_gap'] = df_data['fico'] - (df_data['int.rate'] * 1000)

#ธงเตือนความเสี่ยงสูง: ใช้วงเงินหมุนเวียนเกิน 80%
df_data['high_revol_util'] = (df_data['revol.util'] > 80).astype(int)

#เคยมีประวัติเสีย (รวม delinq + pub.rec เป็นตัวเดียว)
df_data['bad_history_flag'] = ((df_data['delinq.2yrs'] > 0) | (df_data['pub.rec'] > 0)).astype(int)

#credit line อายุยาวเทียบกับอายุ (ใช้ days.with.cr.line / 365)
df_data['years_with_cr_line'] = df_data['days.with.cr.line'] / 365


X = df_data.drop(columns="not.fully.paid")
y = df_data["not.fully.paid"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

corr_matrix = df_data.corr()
target_corr = corr_matrix[["not.fully.paid"]].sort_values(
    by="not.fully.paid", ascending=False
)

print(target_corr)

plt.figure(figsize=(6, 8))
sns.heatmap(target_corr, annot=True, fmt=".3f", cmap="coolwarm", cbar=True)
plt.title("Correlation with not.fully.paid")
plt.show()

# ---------- Scale ONCE, ใช้ร่วมกันทั้งสองโมเดล ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------Logistic Regression ----------
model_1 = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model_1.fit(X_train_scaled, y_train)

y_proba_custom1 = model_1.predict_proba(X_test_scaled)[:, 1]
y_pred_custom1 = (y_proba_custom1 >= 0.50).astype(int)

print("=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_custom1))
print("ROC-AUC :", roc_auc_score(y_test, y_proba_custom1))
print(classification_report(y_test, y_pred_custom1))

cm_log = confusion_matrix(y_test, y_pred_custom1)
ConfusionMatrixDisplay(confusion_matrix=cm_log).plot(cmap="Oranges")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# ---------- Random Forest ----------
classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
classifier.fit(X_train_scaled, y_train)

y_proba_custom2 = classifier.predict_proba(X_test_scaled)[:, 1]
y_pred_custom2 = (y_proba_custom2 >= 0.30).astype(int)

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_custom2))
print("ROC-AUC :", roc_auc_score(y_test, y_proba_custom2))
print(classification_report(y_test, y_pred_custom2))

cm_rf = confusion_matrix(y_test, y_pred_custom2)
ConfusionMatrixDisplay(confusion_matrix=cm_rf).plot(cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.show()
# ----------Linear Regression ----------
linear = LinearRegression()
linear.fit(X_train_scaled, y_train)
y_proba_custom3 = linear.predict(X_test_scaled)
y_pred_custom3 = (y_proba_custom3 >= 0.20).astype(int)
r2 = r2_score(y_test, y_pred_custom3)
print("\n=== Linear Regression ===")
print(f"R2 : {r2}")
print("Accuracy:", accuracy_score(y_test, y_pred_custom3))
print("ROC-AUC :", roc_auc_score(y_test, y_proba_custom3))
print(classification_report(y_test, y_pred_custom3))

cm_lin = confusion_matrix(y_test, y_pred_custom3)
ConfusionMatrixDisplay(confusion_matrix=cm_lin).plot(cmap="Greens")
plt.title("Confusion Matrix - Linear Regression")
plt.show()

# ---------- เทียบ ROC Curve ทั้งสองโมเดลในกราฟเดียว ----------
fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_custom1)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_custom2)

plt.figure(figsize=(7, 6))
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC={roc_auc_score(y_test, y_proba_custom1):.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={roc_auc_score(y_test, y_proba_custom2):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()