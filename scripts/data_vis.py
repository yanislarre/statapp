import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from IPython.display import display


def croissance_pib(pays, df_gdp_smooth_2000):
    # Création de la figure :
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')

    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for idx, country in enumerate(pays):
        df_pays = df_gdp_smooth_2000[df_gdp_smooth_2000['REF_AREA'] == country].sort_values('date')
        ax.plot(df_pays['date'], df_pays['OBS_VALUE'], 
                label=country, linewidth=2.5, marker='o', markersize=4,
                color=couleurs[idx], alpha=0.8)

    # Ajouter une ligne horizontale à PIB = 0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, label='PIB = 0', alpha=0.7)

    # Amélioration du layout
    ax.set_title("Taux de croissance du PIB", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Année", fontsize=12, fontweight='bold')
    ax.set_ylabel("PIB (volume)", fontsize=12, fontweight='bold')

    # Format des dates sur l'axe X
    ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Chaque 5 ans
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()


def reg_m1_m3_salaire(df):
    candidates = ['M1', 'M3', 'Gains horaires']
    results = []
    df_production = df[df['Mesure'] == 'Production volume'].copy()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor('white')

    # Préparer production annuelle agrégée
    # Utiliser la colonne `date` déjà présente dans `df` pour éviter de re-parser
    if 'df_production' in globals():
        df_prod_yearly = df_production.copy()
    else:
        df_prod_yearly = df[df['Mesure'].str.contains('Production', case=False, na=False)].copy()

    if 'date' not in df_prod_yearly.columns:
        df_prod_yearly = df_prod_yearly.merge(df[['TIME_PERIOD','date']].drop_duplicates(), on='TIME_PERIOD', how='left')

    df_prod_yearly = df_prod_yearly[df_prod_yearly['date'].notna()].copy()
    df_prod_yearly['year'] = df_prod_yearly['date'].dt.year

    df_prod_yearly_agg = df_prod_yearly.groupby(['REF_AREA', 'year']).agg({'OBS_VALUE': 'mean'}).reset_index().rename(columns={'OBS_VALUE': 'Production'})

    for i, cand in enumerate(candidates):
        ax = axes[i]
        ax.set_facecolor('#f8f9fa')
        ax.tick_params(axis='both', labelsize=10)

        if cand not in df['Mesure'].unique():
            ax.text(0.5, 0.5, f"{cand} non présent(e)", ha='center', va='center')
            ax.set_title(cand)
            continue

        df_c = df[df['Mesure'] == cand].copy()
        if 'date' not in df_c.columns:
            df_c = df_c.merge(df[['TIME_PERIOD','date']].drop_duplicates(), on='TIME_PERIOD', how='left')
        df_c = df_c[df_c['date'].notna()].copy()
        df_c['year'] = df_c['date'].dt.year
        df_c_agg = df_c.groupby(['REF_AREA', 'year']).agg({'OBS_VALUE': 'mean'}).reset_index().rename(columns={'OBS_VALUE': cand})

        merged = pd.merge(df_prod_yearly_agg, df_c_agg, on=['REF_AREA', 'year'], how='inner').dropna()

        if len(merged) < 30:
            ax.text(0.5, 0.5, f"Pas assez de données (n={len(merged)})", ha='center', va='center')
            ax.set_title(cand)
            continue

        X = merged[[cand]].values
        y = merged['Production'].values

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        coef = float(model.coef_[0])
        intercept = float(model.intercept_)
        corr = merged['Production'].corr(merged[cand])

        results.append({'variable': cand, 'r2': r2, 'coef': coef, 'intercept': intercept, 'corr': corr, 'n': len(merged)})

        # Scatter + droite de régression (matplotlib)
        ax.scatter(merged[cand], merged['Production'], alpha=0.15, s=10, color='#1f77b4')

        # Droite
        x_vals = np.linspace(merged[cand].min(), merged[cand].max(), 100)
        y_line = model.predict(x_vals.reshape(-1, 1))
        ax.plot(x_vals, y_line, color='red', linewidth=2)

        ax.set_title(f"{cand}\nR²={r2:.4f}  corr={corr:.3f}  n={len(merged)}", fontsize=11)
        ax.set_xlabel(cand)
        ax.set_ylabel('Production (volume)')

    plt.tight_layout()
    plt.show()

    # Afficher un tableau récapitulatif
    if results:
        df_results = pd.DataFrame(results).sort_values('r2', ascending=False)
        print('\nRÉCAPITULATIF DES RÉGRESSIONS :')
        display(df_results)
    else:
        print('\nAucun résultat calculé (variables manquantes ou trop peu de données).')
