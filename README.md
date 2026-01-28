# 🏥 Insurance Risk Intelligence & Predictive Pricing

** Démo Live : [Accéder à l'application sur Streamlit Cloud](https://predictive-analysis-g7zjxrbuf79tfb3aolobma.streamlit.app/)**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Vision du Projet
Ce projet dépasse l'analyse exploratoire classique pour proposer une **plateforme de décisionnelle (BI)** dédiée aux assureurs. En s'appuyant sur le dataset *Insurance* (Kaggle), l'application combine **Machine Learning prédictif** et **Scoring de risque actuariel** pour optimiser les politiques tarifaires.

L'objectif est de transformer des données démographiques et comportementales en **recommandations de pricing stratégiques**.

---

##  Fonctionnalités du Dashboard Interactif

L'application Streamlit est structurée en 4 piliers stratégiques :

1.  **📊 Exploration Avancée (EDA)** : Visualisation dynamique de l'impact des facteurs de risque (Tabagisme, BMI, Âge) avec filtres régionaux.
2.  **📈 Modèle Prédictif** : Moteur de régression linéaire estimant les frais médicaux en temps réel selon le profil utilisateur.
3.  **🧠 Insights Automatisés** : Module de *Data Storytelling* générant des synthèses textuelles automatiques pour la direction.
4.  **🎯 Scoring "Risk-Pulse"** : Algorithme propriétaire calculant un score de risque sur 10 et recommandant une action tarifaire (Tarif préférentiel vs Surprime).

---

## 🧠 Méthodologie & Scoring

### 🧪 Algorithme de Risque (Propriétaire)
Le système évalue chaque client sur une échelle de 0 à 10 en pondérant les facteurs critiques identifiés lors de l'analyse :
* **Facteur Majeur** : Tabagisme (**+4 points**)
* **Facteurs Morphologiques** : Obésité (BMI ≥ 30 : **+2 pts**)
* **Facteurs Démographiques** : Âge (≥ 50 ans : **+2 pts**) et situation familiale.



### 📉 Résultats de la Modélisation
L'analyse met en évidence une structure de coût non-linéaire :
* **Médiane Fumeurs** : ~35 000 €
* **Médiane Non-Fumeurs** : ~9 000 €
👉 **Impact** : Le tabagisme multiplie les charges par **3.8x** en moyenne.

| Métrique | Valeur |
| :--- | :--- |
| **Algorithme** | Régression Linéaire |
| **Variable Cible** | Charges Médicales (€) |
| **Validation** | Train/Test Split (80/20) |

---

## 💰 Impact Métier : Optimisation Tarifaire

L'outil traduit le score de risque en décisions de **Smart Pricing** :

| Catégorie | Score | Recommandation Stratégique |
| :--- | :---: | :--- |
| **🟢 FAIBLE** | 0 - 2 | **Tarif préférentiel** (-10% à -20%) |
| **🟡 MOYEN** | 3 - 5 | **Tarif standard** |
| **🟠 ÉLEVÉ** | 6 - 8 | **Surprime modérée** (+15% à +30%) |
| **🔴 CRITIQUE** | 9 - 10 | **Surprime importante** (+35%+) & Suivi médical |



---

## 🛠️ Stack Technique

* **Langage** : Python 3.x
* **Interface** : Streamlit (Web App interactive)
* **Analyse de données** : Pandas, NumPy
* **Visualisation** : Seaborn, Matplotlib, Plotly (interactivité avancée)
* **Machine Learning** : Scikit-learn (Régression Linéaire)

---

## 🎮 Démonstration et Utilisation

Vous pouvez tester la plateforme de deux manières :

### 🌐 Version Cloud (Recommandé)
Accédez instantanément à l'interface interactive ici :  
👉 **[Insurance Risk App - Live Demo](https://predictive-analysis-g7zjxrbuf79tfb3aolobma.streamlit.app/)**

### 💻 Installation Locale
Si vous souhaitez exécuter le projet sur votre machine :
1. **Cloner le répertoire** :
   ```bash
   git clone [https://github.com/Dave-kossi/insurance-risk-intelligence.git](https://github.com/votre-username/insurance-risk-intelligence.git)
