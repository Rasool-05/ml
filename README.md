def find_s_algorithm(examples, target):
    # Initialize most specific hypothesis
    hypothesis = ["ϕ"] * len(examples[0][0])

    for attributes, label in examples:
        if label == target:  # Only consider positive examples
            for i in range(len(attributes)):
                if hypothesis[i] == "ϕ":
                    hypothesis[i] = attributes[i]
                elif hypothesis[i] != attributes[i]:
                    hypothesis[i] = "?"
    return hypothesis

# Dataset: (attributes, target)
dataset = [
    (["Sunny", "Hot", "High", "Weak"], "No"),
    (["Sunny", "Hot", "High", "Strong"], "No"),
    (["Overcast", "Hot", "High", "Weak"], "Yes"),
    (["Rainy", "Mild", "High", "Weak"], "Yes"),
    (["Rainy", "Cool", "Normal", "Weak"], "Yes"),
    (["Rainy", "Cool", "Normal", "Strong"], "No"),
]
final_hypothesis = find_s_algorithm(dataset, "Yes")
print("Final Hypothesis (Find-S):", final_hypothesis)
*
*
def more_general(h1, h2):
    """Check if h1 is more general than h2"""
    more_general_parts = []
    for x, y in zip(h1, h2):
        mg = x == "?" or (x != "ϕ" and (x == y or y == "ϕ"))
        more_general_parts.append(mg)
    return all(more_general_parts)

def candidate_elimination(examples):
    n = len(examples[0][0])
    # Initialize S (most specific) and G (most general)
    S = ["ϕ"] * n
    G = [["?"] * n]

    for attributes, label in examples:
        if label == "Yes":
            # Update S
            for i in range(n):
                if S[i] == "ϕ":
                    S[i] = attributes[i]
                elif S[i] != attributes[i]:
                    S[i] = "?"
            # Remove from G any hypothesis inconsistent with S
            G = [g for g in G if more_general(g, S)]
        else: # Negative example
            new_G = []
            for g in G:
                # If g covers a negative example, specialize g
                # This part of the algorithm logic for G updates can be complex.
                # The original code's structure for generating specializations is kept
                # but properly indented to resolve the SyntaxError.
                for i in range(n):
                    if g[i] == "?":
                        # This condition `if S[i] != "?":` from the original code seems logically questionable for Candidate-Elimination
                        # G-boundary update, but it's preserved as the goal is to fix the SyntaxError.
                        # A correct Candidate Elimination G-boundary update is usually more involved.
                        # Assuming the intention was to generate specializations based on available values.
                        for val in ["Sunny", "Rainy", "Overcast", "Hot", "Mild", "Cool", "High", "Normal", "Weak", "Strong"]:
                            if val != attributes[i]: # Specialize by replacing '?' with a value that is not in the negative example
                                new_hypothesis = g.copy()
                                new_hypothesis[i] = val
                                # Ensure the new hypothesis is more general than S and consistent with negative examples seen so far (implicit here).
                                if more_general(new_hypothesis, S): # This check is also unusual for G-boundary update logic.
                                    new_G.append(new_hypothesis)
            G = new_G
    return S, G

# Dataset
dataset = [
    (["Sunny", "Hot", "High", "Weak"], "No"),
    (["Sunny", "Hot", "High", "Strong"], "No"),
    (["Overcast", "Hot", "High", "Weak"], "Yes"),
    (["Rainy", "Mild", "High", "Weak"], "Yes"),
    (["Rainy", "Cool", "Normal", "Weak"], "Yes"),
    (["Rainy", "Cool", "Normal", "Strong"], "No"),
]

S, G = candidate_elimination(dataset)
print("Final Specific Boundary (S):", S)
print("Final General Boundary (G):", G)
*
*
2
!pip install wittgenstein
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from wittgenstein import RIPPER
# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train RIPPER to classify 'pos_class=0' vs. 'not pos_class=0'
model = RIPPER()
model.fit(X_train, y_train, pos_class=0)
# Print rules
print("Generated Rules:\n", model.ruleset_)
# Predict - model.predict returns True for pos_class, False otherwise
y_pred_bool = model.predict(X_test)
# Convert y_test to a binary boolean array for comparison
y_test_binary_bool = (y_test == 0)
# Calculate accuracy for class 0 prediction
print("Accuracy (Class 0 vs. Others):", accuracy_score(y_test_binary_bool, y_pred_bool))
*
*
3
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bagging
bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
bagging.fit(X_train, y_train)
bag_pred = bagging.predict(X_test)

# Boosting
boosting = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
boosting.fit(X_train, y_train)
boost_pred = boosting.predict(X_test)

print("Bagging Accuracy:", accuracy_score(y_test, bag_pred))
print("Boosting Accuracy:", accuracy_score(y_test, boost_pred))
*
*
4
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate sample dataset
X, _ = make_blobs(n_samples=200, centers=3, random_state=42)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
c='red', marker='X', s=200, label='Centroids')
plt.legend()
plt.show()
*
*
5
!pip install minisom
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
# Load dataset
iris = load_iris()
X = MinMaxScaler().fit_transform(iris.data)
# Initialize SOM
som = MiniSom(x=7, y=7, input_len=4, sigma=1.0, learning_rate=0.5, random_seed=42)
som.random_weights_init(X)
som.train_random(X, 100)
# Visualize SOM
plt.figure(figsize=(7, 7))
for i, x in enumerate(X):
    w = som.winner(x)
    plt.text(w[0], w[1], str(iris.target[i]), color=plt.cm.rainbow(iris.target[i]/2),
             fontdict={'weight': 'bold', 'size': 11})
plt.show()
*
*
6
import numpy as np

# Function to integrate
def f(x):
    return x**2

# Generate random samples
N = 10000
samples = np.random.rand(N)
values = f(samples)

# Monte Carlo estimate
estimate = np.mean(values)
print("Estimated Integral of x^2 from 0 to 1:", estimate)
*
*
7
!pip install pgmpy
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define structure
model = DiscreteBayesianNetwork([('Rain', 'Traffic'), ('Accident', 'Traffic')])

# Define CPDs
cpd_rain = TabularCPD('Rain', 2, [[0.7], [0.3]])
cpd_accident = TabularCPD('Accident', 2, [[0.9], [0.1]])
cpd_traffic = TabularCPD('Traffic', 2,
                            [[0.9, 0.6, 0.7, 0.1],
                             [0.1, 0.4, 0.3, 0.9]],
                            evidence=['Rain', 'Accident'], evidence_card=[2, 2])
model.add_cpds(cpd_rain, cpd_accident, cpd_traffic)

# Inference
infer = VariableElimination(model)
print("P(Traffic=1):", infer.query(['Traffic']).values)
*
*
8
# Inference with evidence
print("P(Traffic=1 | Rain=1):", infer.query(['Traffic'], evidence={'Rain': 1}).values)
print("P(Traffic=1 | Rain=1, Accident=1):", infer.query(['Traffic'], evidence={'Rain': 1, 'Accident': 1}).values)
