# **Travail Pratique : Visualisation des Données (1D, 2D, 3D, Animées)**

## **Objectif**
Explorer les différentes techniques de visualisation pour l'analyse et l'interprétation des données.

## **Prérequis**
Avant de commencer, assurez-vous d'avoir installé les bibliothèques suivantes :
```bash
pip install numpy pandas matplotlib seaborn plotly
```

---

## **📌 Partie 1 : Visualisation en 1D**

### **🔹 Exercice 1 : Histogrammes et Densité**
1. Générer 1000 valeurs aléatoires suivant une loi normale.
2. Tracer un histogramme et une courbe de densité.
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.random.randn(1000)
plt.figure(figsize=(8,5))
sns.histplot(data, bins=30, kde=True, color='blue')
plt.title("Histogramme et densité")
plt.xlabel("Valeurs")
plt.ylabel("Fréquence")
plt.show()
```

### **🔹 Exercice 2 : Boxplot et Outliers**
1. Générer un jeu de données contenant des valeurs extrêmes.
2. Tracer un boxplot pour visualiser les outliers.
```python
sns.boxplot(data=np.append(data, [5, -4, 6, -3]))
plt.title("Boxplot des données")
plt.show()
```

---

## **📌 Partie 2 : Visualisation en 2D**

### **🔹 Exercice 3 : Nuage de Points**
```python
x = np.random.randn(500)
y = 2*x + np.random.randn(500)
plt.scatter(x, y, alpha=0.5, color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Nuage de points")
plt.show()
```

### **🔹 Exercice 4 : Carte de Chaleur des Corrélations**
```python
import pandas as pd
df = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Matrice des corrélations")
plt.show()
```

---

## **📌 Partie 3 : Visualisation en 3D**

### **🔹 Exercice 5 : Nuage de points en 3D**
```python
from mpl_toolkits.mplot3d import Axes3D
x, y = np.random.rand(100), np.random.rand(100)
z = x**2 + y**2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis')
plt.title("Nuage de points 3D")
plt.show()
```

### **🔹 Exercice 6 : Surface 3D**
```python
X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='coolwarm')
plt.title("Surface 3D")
plt.show()
```

---

## **📌 Partie 4 : Animation de Graphiques**

### **🔹 Exercice 7 : Animation d'une fonction sinusoïdale**
```python
import matplotlib.animation as animation
x = np.linspace(0, 4*np.pi, 100)
y = np.sin(x)
fig, ax = plt.subplots()
line, = ax.plot(x, y)
def update(frame):
    line.set_ydata(np.sin(x + frame / 10))
    return line,
ani = animation.FuncAnimation(fig, update, frames=100, interval=50)
plt.show()
```

---

## **📌 Partie 5 : Visualisations interactives avec Plotly**

### **🔹 Exercice 8 : Nuage de points interactif**
```python
import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig.show()
```

### **🔹 Exercice 9 : Surface 3D interactive**
```python
import plotly.graph_objects as go
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
fig.show()
```
