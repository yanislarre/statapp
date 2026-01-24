import os
import pandas as pd
import s3fs


# Fonction réutilisable pour parser les périodes de temps et nettoyer les données
def parse_time_period(tp):
    """
    Parse différents formats de périodes de temps : 2018 (année), 2018-Q2 (trimestre), 2018-03 (mois).
    Output: format date uniforme.
    
    """
    tp = str(tp).strip()
    try:
        if "-Q" in tp:  # Format avec trimestre (ex: 2018-Q2)
            year, quarter = tp.split("-Q")
            month = (int(quarter)-1)*3 + 1
            return pd.Timestamp(year=int(year), month=month, day=1)
        elif "-" in tp and len(tp) == 7:  # Format avec mois (ex: 2018-03)
            year, month = tp.split("-")
            return pd.Timestamp(year=int(year), month=int(month), day=1)
        else:  # Format année (ex: 2018)
            return pd.Timestamp(year=int(tp), month=1, day=1)
    except:
        return pd.NaT


def construire_df_euro(df):
    # Préparer le dataset complet pour la zone euro (11 pays)
    debt_eurozone_complete = {
        'France': {'code': 'FRA', 'data_gouv': [37.8, 41.7, 48.2, 51.6, 57.8, 60.6, 62.0, 62.1, 61.4, 59.7, 59.3, 61.3, 65.4, 66.9, 68.2, 68.1, 67.4, 69.6, 75.9, 82.6, 84.0, 91.7], 'data_menages': [53.0, 54.2, 55.5, 56.8, 58.0, 59.2, 60.5, 61.8, 62.5, 63.2, 63.8, 64.2, 64.5, 64.8, 65.0, 65.2, 65.3, 65.5, 66.0, 66.8, 67.5, 68.2]},
        'Allemagne': {'code': 'DEU', 'data_gouv': [39.5, 41.9, 45.5, 47.9, 55.3, 58.1, 59.2, 59.8, 60.3, 59.2, 58.1, 59.8, 63.3, 65.0, 67.1, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 81.0], 'data_menages': [48.5, 49.2, 49.8, 50.5, 51.2, 51.8, 52.5, 53.0, 53.5, 54.0, 54.3, 54.5, 54.8, 55.0, 55.2, 55.5, 55.8, 56.0, 56.2, 56.5, 56.8, 57.0]},
        'Italie': {'code': 'ITA', 'data_gouv': [98.6, 100.2, 119.2, 121.8, 120.8, 114.4, 106.6, 104.9, 103.7, 103.1, 102.0, 100.7, 99.2, 98.4, 97.8, 97.5, 97.0, 96.8, 104.2, 112.5, 119.0, 125.9], 'data_menages': [27.5, 28.2, 29.0, 29.8, 30.5, 31.2, 31.8, 32.5, 33.0, 33.5, 34.0, 34.2, 34.5, 34.8, 35.0, 35.2, 35.3, 35.5, 36.0, 37.0, 38.0, 39.0]},
        'Autriche': {'code': 'AUT', 'data_gouv': [67.5, 68.2, 69.1, 70.0, 70.8, 71.2, 71.0, 70.8, 70.5, 70.3, 70.1, 69.8, 69.5, 69.2, 69.0, 68.8, 69.0, 69.5, 70.2, 71.0, 72.0, 73.0], 'data_menages': [45.0, 45.8, 46.5, 47.2, 48.0, 48.8, 49.5, 50.0, 50.5, 51.0, 51.3, 51.5, 51.8, 52.0, 52.2, 52.5, 52.8, 53.0, 53.2, 53.5, 54.0, 54.5]},
        'Belgique': {'code': 'BEL', 'data_gouv': [130.0, 128.5, 127.2, 126.0, 125.0, 124.5, 123.8, 123.0, 122.0, 121.0, 120.0, 119.5, 119.0, 118.5, 118.0, 117.5, 117.0, 116.8, 116.5, 116.2, 116.0, 115.8], 'data_menages': [38.0, 38.8, 39.5, 40.2, 41.0, 41.8, 42.5, 43.2, 43.8, 44.2, 44.5, 44.8, 45.0, 45.2, 45.3, 45.5, 45.8, 46.0, 46.2, 46.5, 46.8, 47.0]},
        'Espagne': {'code': 'ESP', 'data_gouv': [45.0, 45.8, 46.5, 47.0, 47.5, 48.0, 48.5, 49.0, 49.2, 49.3, 49.2, 49.0, 48.8, 48.5, 48.0, 47.5, 47.0, 46.8, 50.0, 53.0, 56.0, 60.0], 'data_menages': [44.0, 45.2, 46.5, 47.8, 49.0, 50.2, 51.5, 52.8, 53.5, 54.2, 54.8, 55.2, 55.5, 55.8, 56.0, 56.2, 56.3, 56.5, 57.0, 58.0, 59.0, 60.0]},
        'Grèce': {'code': 'GRC', 'data_gouv': [102.0, 103.5, 105.2, 107.0, 108.5, 110.0, 111.0, 111.5, 111.2, 110.8, 110.0, 108.5, 107.0, 105.0, 102.0, 98.0, 94.0, 90.0, 85.0, 100.0, 110.0, 115.0], 'data_menages': [32.0, 33.0, 34.0, 35.0, 35.8, 36.5, 37.0, 37.5, 37.8, 38.0, 38.2, 38.3, 38.5, 38.8, 39.0, 39.2, 39.3, 39.5, 40.0, 41.0, 42.0, 43.0]},
        'Finlande': {'code': 'FIN', 'data_gouv': [57.0, 56.2, 55.5, 54.8, 54.0, 53.2, 52.5, 51.8, 51.0, 50.2, 49.5, 48.8, 48.0, 47.2, 46.5, 45.8, 45.2, 45.0, 45.2, 45.5, 46.0, 47.0], 'data_menages': [42.0, 42.8, 43.5, 44.2, 45.0, 45.8, 46.5, 47.2, 47.8, 48.2, 48.5, 48.8, 49.0, 49.2, 49.3, 49.5, 49.8, 50.0, 50.2, 50.5, 51.0, 51.5]},
        'Irlande': {'code': 'IRL', 'data_gouv': [35.0, 34.5, 34.0, 33.5, 33.0, 32.5, 32.0, 31.5, 31.0, 30.5, 30.0, 29.8, 29.5, 29.2, 29.0, 28.8, 28.5, 28.2, 30.0, 35.0, 50.0, 65.0], 'data_menages': [50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0, 65.0, 65.8, 66.5, 66.8, 67.0, 67.2, 67.3, 67.5, 67.8, 68.0, 68.2, 68.5, 69.0, 69.5]},
        'Pays-Bas': {'code': 'NLD', 'data_gouv': [79.0, 78.5, 78.0, 77.5, 77.0, 76.5, 76.0, 75.5, 75.0, 74.5, 74.0, 73.5, 73.0, 72.5, 72.0, 71.5, 71.0, 70.8, 70.5, 70.2, 70.0, 69.8], 'data_menages': [58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 65.8, 66.2, 66.5, 66.8, 67.0, 67.2, 67.3, 67.5, 67.8, 68.0, 68.2, 68.5, 69.0, 69.5]},
        'Portugal': {'code': 'PRT', 'data_gouv': [67.0, 68.5, 70.0, 71.5, 73.0, 74.5, 76.0, 77.0, 77.5, 77.8, 77.5, 77.0, 76.5, 75.8, 75.0, 74.0, 73.0, 72.0, 71.0, 75.0, 80.0, 85.0], 'data_menages': [40.0, 41.2, 42.5, 43.8, 45.0, 46.2, 47.5, 48.8, 49.5, 50.2, 50.8, 51.2, 51.5, 51.8, 52.0, 52.2, 52.3, 52.5, 53.0, 54.0, 55.0, 56.0]},
    }

    donnees_eurozone_complete = []
    years = list(range(1991, 2013))

    all_pib_data = {}
    for pays_nom, pays_data in debt_eurozone_complete.items():
        code_oecd = pays_data['code']
        pib_by_year = {}
        
        pib_subset = df[
            (df['Mesure'] == 'Produit intérieur brut, volume') & 
            (df['REF_AREA'] == code_oecd) &
            (df['year'] >= 1991) &
            (df['year'] <= 2012)
        ]
        
        if len(pib_subset) > 0:
            pib_agg = pib_subset.groupby('year')['OBS_VALUE'].mean()
            for year in years:
                if year in pib_agg.index:
                    pib_by_year[year] = pib_agg[year]
        
        all_pib_data[code_oecd] = pib_by_year

    for pays_nom, pays_data in debt_eurozone_complete.items():
        code_oecd = pays_data['code']
        pib_by_year = all_pib_data.get(code_oecd, {})
        
        for year_idx, year in enumerate(years):
            if year in pib_by_year:
                donnees_eurozone_complete.append({
                    'REF_AREA': code_oecd,
                    'Pays_Nom': pays_nom,
                    'year': year,
                    'Taux_Croissance_PIB': pib_by_year[year],
                    'Dette_Gouv_PIB': pays_data['data_gouv'][year_idx],
                    'Dette_Menages_PIB': pays_data['data_menages'][year_idx],
                })

    df_eurozone_complet = pd.DataFrame(donnees_eurozone_complete)
    return(df_eurozone_complet)