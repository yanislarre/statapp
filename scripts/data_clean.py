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
