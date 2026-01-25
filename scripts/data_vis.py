import matplotlib.pyplot as plt
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

    
def dette_gouv(df_debt_gouv):
    # Graphique 1 : Évolution de la dette gouvernementale
    # Restructurer les données pour le graphique

    # Les colonnes (sauf la première) sont les années
    year_columns = [col for col in df_debt_gouv.columns[1:]]

    # Convertir les colonnes d'années en float pour le tri
    try:
        year_columns_int = sorted([(float(col) if isinstance(col, str) else col) for col in year_columns if str(col).replace('.', '').isdigit()])
    except:
        year_columns_int = sorted([col for col in year_columns if isinstance(col, (int, float))])

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')

    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Itérer sur chaque pays (row)
    for idx, row in df_debt_gouv.iterrows():
        country_name = row.iloc[0]  # Premier élément = nom du pays
        
        # Extraire les valeurs pour ce pays
        values = []
        years = []
        
        for col in year_columns_int:
            try:
                val = row[col]
                if pd.notna(val) and val != 'no data' and val != '':
                    # Convertir en float
                    val_float = float(val)
                    values.append(val_float)
                    years.append(col)
            except (ValueError, TypeError):
                pass
        
        # Tracer si on a au moins quelques points
        if len(years) > 1:
            ax.plot(years, values, label=str(country_name), linewidth=2.5, marker='o', markersize=6,
                    color=couleurs[idx % len(couleurs)], alpha=0.8)

    ax.set_title("Évolution de la dette gouvernementale (% du PIB)", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Année", fontsize=12, fontweight='bold')
    ax.set_ylabel("Dette gouvernementale (% du PIB)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def dette_menages(df_debt_household):
    # Graphique 2 : Évolution de la dette des ménages
    # Restructurer les données pour le graphique

    # Les colonnes (sauf la première) sont les années
    year_columns_hh = [col for col in df_debt_household.columns[1:]]

    # Convertir les colonnes d'années en float pour le tri
    try:
        year_columns_hh_int = sorted([(float(col) if isinstance(col, str) else col) for col in year_columns_hh if str(col).replace('.', '').isdigit()])
    except:
        year_columns_hh_int = sorted([col for col in year_columns_hh if isinstance(col, (int, float))])

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')

    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Itérer sur chaque pays (row)
    for idx, row in df_debt_household.iterrows():
        country_name = row.iloc[0]  # Premier élément = nom du pays
        
        # Extraire les valeurs pour ce pays
        values = []
        years = []
        
        for col in year_columns_hh_int:
            try:
                val = row[col]
                if pd.notna(val) and val != 'no data' and val != '':
                    # Convertir en float
                    val_float = float(val)
                    values.append(val_float)
                    years.append(col)
            except (ValueError, TypeError):
                pass
        
        # Tracer si on a au moins quelques points
        if len(years) > 1:
            ax.plot(years, values, label=str(country_name), linewidth=2.5, marker='o', markersize=6,
                    color=couleurs[idx % len(couleurs)], alpha=0.8)

    ax.set_title("Évolution de la dette des ménages (% du PIB)", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Année", fontsize=12, fontweight='bold')
    ax.set_ylabel("Dette des ménages (% du PIB)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def taux_croissance_pib_pay_avec_dette(df_pib_agg, pays_dette):
    # Créer la figure
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')

    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    # Tracer chaque pays
    for idx, (pays_nom, code_pays) in enumerate(pays_dette.items()):
        df_pays = df_pib_agg[df_pib_agg['REF_AREA'] == code_pays].sort_values('date')
        
        if len(df_pays) > 0:
            ax.plot(df_pays['date'], df_pays['OBS_VALUE'], 
                    label=pays_nom, linewidth=2.5, marker='o', markersize=5,
                    color=couleurs[idx % len(couleurs)], alpha=0.8)
        else:
            print(f"⚠️  Pas de données de PIB pour {pays_nom} ({code_pays})")

    # Ajouter une ligne horizontale à 0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    ax.set_title("Taux de croissance du PIB - Pays avec données de dette", 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Année", fontsize=12, fontweight='bold')
    ax.set_ylabel("Taux de croissance du PIB (%)", fontsize=12, fontweight='bold')

    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()


def comparaison_pays_dette_pib(country_names, df_debt_gouv, df_debt_household):
    # Créer une figure avec 4 sous-graphiques (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.patch.set_facecolor('white')

    for idx, (country_label, country_name) in enumerate(country_names.items()):
        ax = axes[idx]
        ax.set_facecolor('#f8f9fa')
        
        # Récupérer les données de dette gouvernementale
        debt_gouv_row = df_debt_gouv[df_debt_gouv.iloc[:, 0].str.contains(country_name, case=False, na=False)]
        debt_household_row = df_debt_household[df_debt_household.iloc[:, 0].str.contains(country_name, case=False, na=False)]
        
        plotted = False
        
        # Tracer la dette gouvernementale
        if len(debt_gouv_row) > 0:
            row = debt_gouv_row.iloc[0]
            years_gouv = []
            values_gouv = []
            
            for col in df_debt_gouv.columns[1:]:
                try:
                    year = float(col) if str(col).replace('.', '').isdigit() else None
                    if year and pd.notna(row[col]) and row[col] != 'no data' and row[col] != '':
                        years_gouv.append(int(year))
                        values_gouv.append(float(row[col]))
                except (ValueError, TypeError):
                    pass
            
            if len(years_gouv) > 1:
                ax.plot(years_gouv, values_gouv, 
                        marker='o', 
                        linewidth=2.5, 
                        label='Dette gouvernementale',
                        color='#2E86AB',
                        markersize=5,
                        alpha=0.8)
                plotted = True
        
        # Tracer la dette des ménages
        if len(debt_household_row) > 0:
            row = debt_household_row.iloc[0]
            years_hh = []
            values_hh = []
            
            for col in df_debt_household.columns[1:]:
                try:
                    year = float(col) if str(col).replace('.', '').isdigit() else None
                    if year and pd.notna(row[col]) and row[col] != 'no data' and row[col] != '':
                        years_hh.append(int(year))
                        values_hh.append(float(row[col]))
                except (ValueError, TypeError):
                    pass
            
            if len(years_hh) > 1:
                ax.plot(years_hh, values_hh, 
                        marker='s', 
                        linewidth=2.5, 
                        label='Dette privée (ménages)',
                        color='#A23B72',
                        markersize=5,
                        alpha=0.8)
                plotted = True
        
        # Personnalisation du graphique
        ax.set_xlabel('Année', fontsize=12, fontweight='bold')
        ax.set_ylabel('Dette (% du PIB)', fontsize=12, fontweight='bold')
        ax.set_title(f'{country_label}', fontsize=14, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=10)
        
        if not plotted:
            ax.text(0.5, 0.5, 'Données non disponibles', 
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)

    plt.suptitle('Évolution de la dette gouvernementale et de la dette privée (ménages) pour 4 pays', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def pib_dette_gouv_france(df_france_analyse):
    # Graphique 1 : Évolution temporelle - Dette gouvernementale vs Taux de croissance
    fig, ax1 = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('#f8f9fa')

    # Axe gauche : Taux de croissance du PIB
    color1 = '#E63946'
    ax1.set_xlabel('Année', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Taux de croissance du PIB (%)', fontsize=12, fontweight='bold', color=color1)
    ax1.plot(df_france_analyse['year'], df_france_analyse['Taux_Croissance_PIB'], 
            linewidth=2.5, marker='o', markersize=6, color=color1, alpha=0.8, label='Croissance PIB')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Axe droit : Dette gouvernementale (% du PIB)
    ax2 = ax1.twinx()
    color2 = '#2E86AB'
    ax2.set_ylabel('Dette gouvernementale (% du PIB)', fontsize=12, fontweight='bold', color=color2)
    ax2.plot(df_france_analyse['year'], df_france_analyse['Dette_Gouv_PIB'], 
            linewidth=2.5, marker='s', markersize=6, color=color2, alpha=0.8, label='Dette gouvernementale')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Titre et légende
    fig.suptitle('France : Taux de croissance du PIB et dette gouvernementale', 
                fontsize=16, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left', framealpha=0.95)

    plt.tight_layout()
    plt.show()


def pib_dette_menages_france(df_france_analyse):
    # Graphique 2 : Évolution temporelle - Dette des ménages vs Taux de croissance
    fig, ax1 = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('#f8f9fa')

    # Axe gauche : Taux de croissance du PIB
    color1 = '#E63946'
    ax1.set_xlabel('Année', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Taux de croissance du PIB (%)', fontsize=12, fontweight='bold', color=color1)
    ax1.plot(df_france_analyse['year'], df_france_analyse['Taux_Croissance_PIB'], 
            linewidth=2.5, marker='o', markersize=6, color=color1, alpha=0.8, label='Croissance PIB')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Axe droit : Dette des ménages (% du PIB)
    ax2 = ax1.twinx()
    color2 = '#A23B72'
    ax2.set_ylabel('Dette des ménages (% du PIB)', fontsize=12, fontweight='bold', color=color2)
    ax2.plot(df_france_analyse['year'], df_france_analyse['Dette_Menages_PIB'], 
            linewidth=2.5, marker='s', markersize=6, color=color2, alpha=0.8, label='Dette ménages')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Titre et légende
    fig.suptitle('France : Taux de croissance du PIB vs Dette des ménages', 
                fontsize=16, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left', framealpha=0.95)

    plt.tight_layout()
    plt.show()


def reg_dette_pib_france(df_france_analyse):
    # Graphique 3 : Nuages de points avec régression linéaire
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')

    # Sous-graphique 1 : Dette gouvernementale vs Croissance PIB
    ax1 = axes[0]
    ax1.set_facecolor('#f8f9fa')

    X_gouv = df_france_analyse[['Dette_Gouv_PIB']].values
    y_pib = df_france_analyse['Taux_Croissance_PIB'].values

    # Régression linéaire
    model_gouv = LinearRegression()
    model_gouv.fit(X_gouv, y_pib)
    y_pred_gouv = model_gouv.predict(X_gouv)
    r2_gouv = r2_score(y_pib, y_pred_gouv)
    coef_gouv = model_gouv.coef_[0]
    intercept_gouv = model_gouv.intercept_

    # Nuage de points
    ax1.scatter(X_gouv, y_pib, s=80, alpha=0.6, color='#2E86AB', edgecolors='black', linewidth=1)

    # Droite de régression
    x_line = np.linspace(X_gouv.min(), X_gouv.max(), 100)
    y_line_gouv = model_gouv.predict(x_line.reshape(-1, 1))
    ax1.plot(x_line, y_line_gouv, color='red', linewidth=3, 
            label=f'y = {intercept_gouv:.2f} + {coef_gouv:.4f}x\nR² = {r2_gouv:.4f}')

    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Dette gouvernementale (% du PIB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Taux de croissance du PIB (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Gouvernement - France', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Sous-graphique 2 : Dette des ménages vs Croissance PIB
    ax2 = axes[1]
    ax2.set_facecolor('#f8f9fa')

    X_menages = df_france_analyse[['Dette_Menages_PIB']].values

    # Régression linéaire
    model_menages = LinearRegression()
    model_menages.fit(X_menages, y_pib)
    y_pred_menages = model_menages.predict(X_menages)
    r2_menages = r2_score(y_pib, y_pred_menages)
    coef_menages = model_menages.coef_[0]
    intercept_menages = model_menages.intercept_

    # Nuage de points
    ax2.scatter(X_menages, y_pib, s=80, alpha=0.6, color='#A23B72', edgecolors='black', linewidth=1)

    # Droite de régression
    x_line2 = np.linspace(X_menages.min(), X_menages.max(), 100)
    y_line_menages = model_menages.predict(x_line2.reshape(-1, 1))
    ax2.plot(x_line2, y_line_menages, color='red', linewidth=3, 
            label=f'y = {intercept_menages:.2f} + {coef_menages:.4f}x\nR² = {r2_menages:.4f}')

    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Dette des ménages (% du PIB)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Taux de croissance du PIB (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Ménages - France', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best', framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()

    # Afficher les résultats des régressions
    print("\n" + "=" * 80)
    print("RÉSULTATS DES RÉGRESSIONS LINÉAIRES - FRANCE")
    print("=" * 80)
    print("\n1. DETTE GOUVERNEMENTALE vs CROISSANCE DU PIB :")
    print(f"   Équation : Croissance PIB = {intercept_gouv:.2f} + {coef_gouv:.4f} × Dette_Gouv")
    print(f"   Coefficient de corrélation (R²) : {r2_gouv:.4f}")
    print(f"   Interprétation : Chaque augmentation de 1% de la dette gouvernementale (% PIB)")
    print(f"                    est associée à une variation de {coef_gouv:.4f}% de la croissance du PIB")

    print("\n2. DETTE DES MÉNAGES vs CROISSANCE DU PIB :")
    print(f"   Équation : Croissance PIB = {intercept_menages:.2f} + {coef_menages:.4f} × Dette_Ménages")
    print(f"   Coefficient de corrélation (R²) : {r2_menages:.4f}")
    print(f"   Interprétation : Chaque augmentation de 1% de la dette des ménages (% PIB)")
    print(f"                    est associée à une variation de {coef_menages:.4f}% de la croissance du PIB")
    print("\n" + "=" * 80)


def reg_multi_pays(df_eurozone_complet):
    # Régressions pour la zone euro étendue - Avec tous les pays disponibles
    print("\n" + "=" * 100)
    print("RÉGRESSIONS ZONE EURO ÉTENDUE (1991-2012)")
    print("=" * 100)

    # Vérifier quels pays sont effectivement dans le dataset
    pays_disponibles_eurozone = df_eurozone_complet['REF_AREA'].unique()
    print(f"\nPays dans le dataset : {len(pays_disponibles_eurozone)}")
    print(f"  {', '.join(sorted(df_eurozone_complet['Pays_Nom'].unique()))}")

    # Préparation des données pour la régression
    X_gouv_euro = df_eurozone_complet[['Dette_Gouv_PIB']].values
    y_pib_euro = df_eurozone_complet['Taux_Croissance_PIB'].values

    # Régression
    model_gouv_euro = LinearRegression()
    model_gouv_euro.fit(X_gouv_euro, y_pib_euro)
    y_pred_gouv_euro = model_gouv_euro.predict(X_gouv_euro)
    r2_gouv_euro = r2_score(y_pib_euro, y_pred_gouv_euro)
    coef_gouv_euro = model_gouv_euro.coef_[0]
    intercept_gouv_euro = model_gouv_euro.intercept_

    print(f"\nRégression - Dette gouvernementale vs Croissance PIB")
    print(f"  Nombre d'observations : {len(df_eurozone_complet)}")
    print(f"  Nombre de pays : {df_eurozone_complet['REF_AREA'].nunique()}")
    print(f"  Équation : Croissance = {intercept_gouv_euro:.4f} + {coef_gouv_euro:.4f} × Dette_Gouv")
    print(f"  R² : {r2_gouv_euro:.4f}")
    print(f"  Pente (coefficient) : {coef_gouv_euro:.4f}")

    # Créer les graphiques
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_facecolor('#f8f9fa')

    # Tracer tous les points (tous les pays fusionnés)
    ax.scatter(X_gouv_euro, y_pib_euro, s=100, alpha=0.65, color='steelblue', 
            edgecolors='black', linewidth=0.8, zorder=2)

    # Droite de régression
    x_line_euro = np.linspace(X_gouv_euro.min() - 5, X_gouv_euro.max() + 5, 100)
    y_line_euro = model_gouv_euro.predict(x_line_euro.reshape(-1, 1))
    ax.plot(x_line_euro, y_line_euro, color='red', linewidth=3.5, linestyle='--',
        label=f'Régression: y = {intercept_gouv_euro:.2f} + {coef_gouv_euro:.4f}x', zorder=3)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Dette gouvernementale (% du PIB)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Taux de croissance du PIB (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Gouvernement - zone euro ({df_eurozone_complet["REF_AREA"].nunique()} pays, 1991-2012)\nR² = {r2_gouv_euro:.4f}', 
                fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()

    # Graphique - Dette des ménages vs Croissance PIB
    X_menages_euro = df_eurozone_complet[['Dette_Menages_PIB']].dropna()
    if len(X_menages_euro) > 0:
        y_pib_menages_euro = df_eurozone_complet.loc[X_menages_euro.index, 'Taux_Croissance_PIB'].values
        
        model_menages_euro = LinearRegression()
        model_menages_euro.fit(X_menages_euro, y_pib_menages_euro)
        y_pred_menages_euro = model_menages_euro.predict(X_menages_euro)
        r2_menages_euro = r2_score(y_pib_menages_euro, y_pred_menages_euro)
        coef_menages_euro = model_menages_euro.coef_[0]
        intercept_menages_euro = model_menages_euro.intercept_
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.set_facecolor('#f8f9fa')
        
        ax.scatter(X_menages_euro, y_pib_menages_euro, s=100, alpha=0.65, color='darkgreen', 
                edgecolors='black', linewidth=0.8, zorder=2)
        
        x_line_menages = np.linspace(X_menages_euro.min().values[0] - 5, X_menages_euro.max().values[0] + 5, 100)
        y_line_menages = model_menages_euro.predict(x_line_menages.reshape(-1, 1))
        ax.plot(x_line_menages, y_line_menages, color='red', linewidth=3.5, linestyle='--',
            label=f'Régression: y = {intercept_menages_euro:.2f} + {coef_menages_euro:.4f}x', zorder=3)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Dette des ménages (% du PIB)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Taux de croissance du PIB (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'Ménages - zone euro (11 pays, 1991-2012)\nR² = {r2_menages_euro:.4f}', 
                    fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.show()

    
def pib_dette_gouv_interet_france(df_france_analyse):
    fig, ax1 = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('#f8f9fa')

    # ─────────────────────────────────────────
    # Axe gauche : Taux de croissance du PIB
    # ─────────────────────────────────────────
    color1 = '#E63946'
    ax1.set_xlabel('Année', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Taux de croissance du PIB (%)', fontsize=12, fontweight='bold', color=color1)

    ax1.plot(
        df_france_analyse['year'],
        df_france_analyse['Taux_Croissance_PIB'],
        linewidth=2.5,
        marker='o',
        markersize=6,
        color=color1,
        alpha=0.8,
        label='Croissance du PIB'
    )

    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # ─────────────────────────────────────────
    # Axe droit : Dette + Taux d’intérêt long terme
    # ─────────────────────────────────────────
    ax2 = ax1.twinx()
    color2 = '#2E86AB'
    color3 = '#2A9D8F'

    ax2.set_ylabel('Dette (% PIB) / Taux d’intérêt (%)', fontsize=12, fontweight='bold')

    # Dette gouvernementale
    ax2.plot(
        df_france_analyse['year'],
        df_france_analyse['Dette_Gouv_PIB'],
        linewidth=2.5,
        marker='s',
        markersize=6,
        color=color2,
        alpha=0.8,
        label='Dette gouvernementale (% PIB)'
    )

    # Taux d’intérêt à long terme
    ax2.plot(
        df_france_analyse['year'],
        df_france_analyse['Taux_Interet_Long_Terme'],
        linewidth=2.5,
        linestyle='--',
        marker='^',
        markersize=6,
        color=color3,
        alpha=0.85,
        label='Taux d’intérêt à long terme'
    )

    ax2.tick_params(axis='y')

    # ─────────────────────────────────────────
    # Titre & légende
    # ─────────────────────────────────────────
    fig.suptitle(
        'France : Croissance du PIB, dette publique et taux d’intérêt à long terme',
        fontsize=16,
        fontweight='bold'
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        fontsize=11,
        loc='upper left',
        framealpha=0.95
    )

    plt.tight_layout()
    plt.show()
