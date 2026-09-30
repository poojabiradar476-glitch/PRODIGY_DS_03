import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Generate Bank Marketing Dataset Structure
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'age': np.random.randint(18, 70, size=n),
    'job': np.random.choice(['admin.', 'blue-collar', 'technician', 'services'], size=n),
    'marital': np.random.choice(['married', 'single', 'divorced'], size=n),
    'balance': np.random.randint(-500, 5000, size=n),
    'housing': np.random.choice(['yes', 'no'], size=n),
    'deposit': np.random.choice(['yes', 'no'], size=n, p=[0.3, 0.7])
})

# Preprocessing
df_encoded = pd.get_dummies(df.drop(columns=['deposit']), drop_first=True)
X = df_encoded
y = df['deposit'].map({'yes': 1, 'no': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_state=42, test_size=0.2)

# Model Training
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize Tree
plt.figure(figsize=(16, 8))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, fontsize=10)
plt.savefig('decision_tree.png')
plt.show()
print("Task 3 Completed successfully.")
