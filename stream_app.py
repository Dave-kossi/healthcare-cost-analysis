import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# ------------------------------
# Chargement des données
# ------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

df = load_data()

# ------------------------------
# FONCTION DE SCORING DE RISQUE
# ------------------------------
def calculer_score_risque(age, bmi, smoker, children):
    """Calcule le score de risque client (0-10)"""
    score = 0
    
    # Facteurs critiques
    if smoker == "yes": 
        score += 4
    if age >= 50: 
        score += 2
    if bmi >= 30: 
        score += 2
    
    # Facteurs secondaires
    if age >= 40: 
        score += 1
    if bmi >= 25: 
        score += 1
    if children >= 2: 
        score += 1
    
    return min(score, 10)

def categoriser_risque(score):
    """Catégorise le score en niveau de risque"""
    if score <= 2:
        return "🟢 FAIBLE", "low"
    elif score <= 5:
        return "🟡 MOYEN", "medium"
    elif score <= 8:
        return "🟠 ÉLEVÉ", "high"
    else:
        return "🔴 TRÈS ÉLEVÉ", "very_high"

def get_recommandation_pricing(categorie):
    """Retourne les recommandations de pricing par catégorie"""
    recommendations = {
        "low": "✅ Tarif préférentiel (-10% à -20%)",
        "medium": "📗 Tarif standard (0% à +10%)", 
        "high": "⚠️ Surprime modérée (+15% à +30%)",
        "very_high": "🚨 Surprime importante (+35% à +60%)"
    }
    return recommendations.get(categorie, "📊 Analyse requise")

# ------------------------------
# Configuration de la page
# ------------------------------
st.set_page_config(page_title="Insurance Risk Intelligence", page_icon="🏥", layout="wide")

st.title("🏥 Insurance Risk Intelligence")
st.markdown("### Plateforme de scoring client et optimisation tarifaire")

st.write("""
Cette application analyse les **facteurs de risque santé** et optimise la **tarification assurance** grâce au machine learning.  
Elle combine **scoring client avancé**, **visualisation interactive** et **recommandations stratégiques** pour une gestion optimale du portefeuille clients.
""")

# ------------------------------
# Navigation par onglets
# ------------------------------
tabs = st.tabs(["📊 Exploration", "📈 Modèle prédictif", "🧠 Insights automatiques", "🎯 Scoring Risque Client"])

# ==========================================================
# 1️⃣ Onglet Exploration edité
# ==========================================================
with tabs[0]:
    st.header("📊 Exploration des variables")

    regions = ["Toutes les régions"] + sorted(df["region"].unique().tolist())
    region = st.selectbox("🌍 Sélectionnez une région :", regions, key="explore_region")

    # Gestion du filtre global
    if region == "Toutes les régions":
        filtered_df = df.copy()
    else:
        filtered_df = df[df["region"] == region]

    # --- Boxplot tabagisme
    st.subheader(f"🚬 Impact du tabagisme sur les frais médicaux ({region})")

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=filtered_df, x="smoker", y="charges", palette="coolwarm", ax=ax1)
    st.pyplot(fig1)

    median_smoker = filtered_df[filtered_df["smoker"]=="yes"]["charges"].median()
    median_non = filtered_df[filtered_df["smoker"]=="no"]["charges"].median()
    ratio = median_smoker / median_non if median_non > 0 else 0

    st.markdown(f"""
    💬 **Observation :**  
    - Médiane fumeurs : **{median_smoker:,.0f} €**  
    - Médiane non-fumeurs : **{median_non:,.0f} €**  
    👉 Les fumeurs paient environ **{ratio:.1f}× plus** en frais médicaux.
    """)

    # --- Corrélation âge / frais
    st.subheader("🎂 Relation entre l'âge et les frais médicaux")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=filtered_df, x="age", y="charges", hue="smoker", alpha=0.7, palette="coolwarm", ax=ax2)
    st.pyplot(fig2)

    cor_age = filtered_df["age"].corr(filtered_df["charges"])
    st.markdown(f"""
    💬 **Analyse :**  
    Corrélation âge/frais : **{cor_age:.2f}**  
    👉 Les frais augmentent avec l'âge, surtout chez les fumeurs.
    """)

    # --- Corrélation BMI / frais
    st.subheader("⚖️ Relation entre le BMI et les frais médicaux")

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=filtered_df, x="bmi", y="charges", hue="smoker", alpha=0.7, palette="coolwarm", ax=ax3)
    st.pyplot(fig3)

    cor_bmi = filtered_df["bmi"].corr(filtered_df["charges"])
    st.markdown(f"""
    💬 **Analyse :**  
    Corrélation BMI/frais : **{cor_bmi:.2f}**  
    👉 Un BMI élevé (>30) tend à augmenter les coûts, mais le **tabagisme reste le facteur dominant**.
    """)

# ==========================================================
# 2️⃣ Onglet Modèle prédictif (édité)
# ==========================================================
with tabs[1]:
    st.header("📈 Modèle de Régression Linéaire")

    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop("charges", axis=1)
    y = df_encoded["charges"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    st.write("### 🧮 Entrez les paramètres pour estimer les frais médicaux :")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Âge :", 18, 64, 30)
        children = st.selectbox("Nombre d'enfants :", [0, 1, 2, 3, 4, 5])
    with col2:
        bmi = st.slider("BMI :", 15.0, 50.0, 25.0)
        smoker = st.selectbox("Fumeur :", ["yes", "no"])
    with col3:
        sex = st.selectbox("Sexe :", ["male", "female"])
        region_input = st.selectbox("Région :", sorted(df["region"].unique().tolist()))

    sample = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [1 if sex == "male" else 0],
        "smoker_yes": [1 if smoker == "yes" else 0],
        "region_northwest": [1 if region_input == "northwest" else 0],
        "region_southeast": [1 if region_input == "southeast" else 0],
        "region_southwest": [1 if region_input == "southwest" else 0]
    })

    prediction = model.predict(sample)[0]
    st.success(f"💰 **Estimation des frais médicaux : {prediction:,.2f} €**")

    # Commentaires dynamiques
    if smoker == "yes":
        st.info("🚭 Le statut de fumeur augmente fortement les coûts médicaux.")
    if bmi > 30:
        st.warning("⚠️ Un BMI supérieur à 30 accroît significativement les dépenses médicales.")
    if age > 50:
        st.info("📈 L'âge avancé est associé à une hausse des frais médicaux moyens.")

    st.caption("🔧 Modèle linéaire en cours de développement — à des fins éducatives.")

# ==========================================================
# 3️⃣ Onglet Insights automatiques (édité)
# ==========================================================
with tabs[2]:
    st.header("🧠 Synthèse automatique des insights")

    st.markdown("""
    Cette section génère une **interprétation automatique** des tendances observées dans les données.  
    Idéale pour le **data storytelling** et la **présentation à la direction**.
    """)

    st.markdown("### 📋 Résumé global :")
    st.write(f"- **Corrélation âge/frais :** {cor_age:.2f}")
    st.write(f"- **Corrélation BMI/frais :** {cor_bmi:.2f}")
    st.write(f"- **Impact du tabagisme :** environ {ratio:.1f}× plus de dépenses pour les fumeurs.")
    st.write("- **Différences régionales :** faibles variations, tendance générale similaire.")

    st.markdown("---")
    st.subheader("🧩 Interprétation globale :")

    interpretation = f"""
    > Le **tabagisme** demeure le facteur dominant des coûts de santé, amplifiant les dépenses d'un facteur 3 à 4.  
    > Le **BMI** et l'**âge** jouent un rôle secondaire mais significatif dans l'augmentation des frais.  
    > Globalement, les **tendances régionales restent cohérentes**, ce qui montre que les effets sont 
    davantage liés au comportement qu'à la localisation.  
    > Ces résultats soutiennent des politiques de **prévention santé** et d'**ajustement du risque assurantiel**.
    """

    st.markdown(interpretation)

    st.success("✅ Interprétation automatique générée à partir des tendances du dataset.")
    #st.caption("Analyse réalisée par **Kossi Noumagno — Data Analyst | Machine Learning & Data Storytelling**")

# ==========================================================
# 4️⃣ NOUVEL ONGLET : Scoring Risque Client
# ==========================================================
with tabs[3]:
    st.header("🎯 Scoring de Risque Client")
    st.markdown("""
    **Évaluez le niveau de risque de vos clients** pour optimiser la tarification et la gestion de portefeuille.
    Le scoring combine l'âge, le BMI, le tabagisme et le nombre d'enfants.
    """)
    
    # Interface de saisie
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Informations client")
        age_risk = st.slider("Âge du client :", 18, 70, 35, key="risk_age")
        bmi_risk = st.slider("BMI du client :", 15.0, 50.0, 25.0, key="risk_bmi")
        
    with col2:
        st.subheader("🧬 Comportements santé")
        smoker_risk = st.radio("Statut tabagique :", ["no", "yes"], key="risk_smoker")
        children_risk = st.selectbox("Nombre d'enfants :", [0, 1, 2, 3, 4, 5], key="risk_children")
    
    # Calcul du score
    if st.button("🎯 Calculer le Score de Risque", type="primary"):
        score = calculer_score_risque(age_risk, bmi_risk, smoker_risk, children_risk)
        categorie, niveau = categoriser_risque(score)
        recommandation = get_recommandation_pricing(niveau)
        
        # Affichage des résultats
        st.markdown("---")
        
        # Métriques
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Score de Risque", f"{score}/10")
        
        with col2:
            st.metric("🎯 Catégorie", categorie)
            
        with col3:
            st.metric("💰 Recommandation", recommandation.split(" ")[0])
        
        # Détail du scoring
        st.subheader("🔍 Détail du calcul du score")
        
        details = []
        if smoker_risk == "yes":
            details.append("🚭 **Fumeur** : +4 points (risque majeur)")
        if age_risk >= 50:
            details.append("🎂 **Âge ≥ 50 ans** : +2 points")
        if bmi_risk >= 30:
            details.append("⚖️ **BMI ≥ 30** : +2 points (obésité)")
        if age_risk >= 40:
            details.append("📈 **Âge ≥ 40 ans** : +1 point")
        if bmi_risk >= 25:
            details.append("⚖️ **BMI ≥ 25** : +1 point (surpoids)")
        if children_risk >= 2:
            details.append("👨‍👩‍👧‍👦 **≥ 2 enfants** : +1 point")
        
        for detail in details:
            st.write(f"- {detail}")
        
        # Recommandation détaillée
        st.subheader("💡 Recommandation stratégique")
        st.info(recommandation)
        
        # Justification basée sur les facteurs
        st.subheader("🎯 Plan d'action recommandé")
        if niveau == "low":
            st.success("**Stratégie :** Fidélisation et tarifs attractifs. Client très rentable.")
        elif niveau == "medium":
            st.info("**Stratégie :** Surveillance standard. Client à profitabilité moyenne.")
        elif niveau == "high":
            st.warning("**Stratégie :** Surveillance accrue + programmes de prévention. Client à risque modéré.")
        else:
            st.error("**Stratégie :** Surprime significative + suivi médical renforcé. Client à très haut risque.")
    
    # Analyse du portefeuille global
    st.markdown("---")
    st.subheader("📈 Analyse du Portefeuille Client")
    
    # Filtres pour l'analyse de portefeuille
    st.write("**Filtres pour l'analyse :**")
    col_filter1, col_filter2 = st.columns(2)
    
    with col_filter1:
        region_portfolio = st.selectbox(
            "🌍 Région à analyser :",
            ["Toutes les régions"] + sorted(df["region"].unique().tolist()),
            key="portfolio_region"
        )
    
    with col_filter2:
        smoker_filter = st.selectbox(
            "🚬 Filtre tabagisme :",
            ["Tous", "Fumeurs uniquement", "Non-fumeurs uniquement"],
            key="portfolio_smoker"
        )
    
    # Application des filtres
    filtered_portfolio = df.copy()
    
    if region_portfolio != "Toutes les régions":
        filtered_portfolio = filtered_portfolio[filtered_portfolio["region"] == region_portfolio]
    
    if smoker_filter == "Fumeurs uniquement":
        filtered_portfolio = filtered_portfolio[filtered_portfolio["smoker"] == "yes"]
    elif smoker_filter == "Non-fumeurs uniquement":
        filtered_portfolio = filtered_portfolio[filtered_portfolio["smoker"] == "no"]
    
    if st.checkbox("🔄 Calculer la répartition des risques sur le portefeuille filtré"):
        # Application du scoring au dataset filtré
        filtered_portfolio['score_risque'] = filtered_portfolio.apply(
            lambda row: calculer_score_risque(row['age'], row['bmi'], row['smoker'], row['children']), 
            axis=1
        )
        filtered_portfolio['categorie_risque'] = filtered_portfolio['score_risque'].apply(lambda x: categoriser_risque(x)[0])
        
        # Visualisation
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                filtered_portfolio, 
                names='categorie_risque', 
                title=f'📊 Répartition des Risques - {region_portfolio}'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Graphique des coûts moyens par catégorie
            couts_moyens = filtered_portfolio.groupby('categorie_risque')['charges'].mean().sort_values()
            fig = px.bar(
                x=couts_moyens.index, 
                y=couts_moyens.values,
                title=f'💰 Coûts Moyens par Risque - {region_portfolio}',
                labels={'x': 'Catégorie de Risque', 'y': 'Coût Moyen (€)'},
                color=couts_moyens.index,
                color_discrete_map={
                    '🟢 FAIBLE': 'green',
                    '🟡 MOYEN': 'orange', 
                    '🟠 ÉLEVÉ': 'red',
                    '🔴 TRÈS ÉLEVÉ': 'darkred'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques dynamiques
        st.subheader("📋 Statistiques du Portefeuille Filtré")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risque_eleve = len(filtered_portfolio[filtered_portfolio['score_risque'] >= 6])
            total_clients = len(filtered_portfolio)
            st.metric(
                "🚨 Clients Risque Élevé+", 
                f"{risque_eleve}/{total_clients}",
                f"{risque_eleve/total_clients*100:.1f}%"
            )
        
        with col2:
            cout_moyen_risque = filtered_portfolio[filtered_portfolio['score_risque'] >= 6]['charges'].mean()
            cout_moyen_total = filtered_portfolio['charges'].mean()
            st.metric(
                "💸 Coût Moyen Risque Élevé", 
                f"{cout_moyen_risque:,.0f} €",
                f"{(cout_moyen_risque/cout_moyen_total-1)*100:+.1f}% vs moyenne"
            )
        
        with col3:
            part_couts_risque = filtered_portfolio[filtered_portfolio['score_risque'] >= 6]['charges'].sum() / filtered_portfolio['charges'].sum() * 100
            st.metric(
                "📈 Part des Coûts Risque Élevé", 
                f"{part_couts_risque:.1f}%"
            )
        
        # Tableau détaillé
        st.subheader("📊 Détail par Catégorie de Risque")
        detail_par_categorie = filtered_portfolio.groupby('categorie_risque').agg({
            'charges': ['count', 'mean', 'sum'],
            'age': 'mean',
            'bmi': 'mean'
        }).round(1)
        
        detail_par_categorie.columns = ['Nb Clients', 'Coût Moyen', 'Coût Total', 'Âge Moyen', 'BMI Moyen']
        detail_par_categorie['Part Clients'] = (detail_par_categorie['Nb Clients'] / total_clients * 100).round(1)
        detail_par_categorie['Part Coûts'] = (detail_par_categorie['Coût Total'] / filtered_portfolio['charges'].sum() * 100).round(1)
        
        st.dataframe(detail_par_categorie)

# Section : Historique du Dataset
st.sidebar.markdown("### 🗂️ Historique du Dataset")
if st.sidebar.checkbox("Afficher l'historique des données"):
    st.markdown("""
    ### 📘 Historique du Dataset - *Insurance Charges (Kaggle)*  
    Le dataset **Insurance** provient de la plateforme [Kaggle](https://www.kaggle.com/).  
    Il contient des informations sur les **frais médicaux individuels** en fonction de variables démographiques et comportementales :
    
    - 👤 **age** : âge du bénéficiaire de l'assurance  
    - ⚖️ **bmi** : indice de masse corporelle  
    - 🧒 **children** : nombre d'enfants à charge  
    - 🚬 **smoker** : indique si la personne fume  
    - 🌍 **region** : région de résidence  
    - 💰 **charges** : frais médicaux facturés  

    **Objectif** : comprendre et modéliser les facteurs influençant le coût des soins de santé afin d'optimiser la tarification des assurances.  
    """)
