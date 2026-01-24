import os
import s3fs
import pandas as pd


def get_cloud_csv(filename, sep=","):
    """
    Charge un fichier CSV depuis S3 et retourne un DataFrame.

    Paramètres:
    -----------
    filename : str
        Nom du fichier à charger (avec ou sans extension .csv)
    sep : str, optional
        Séparateur du fichier CSV (par défaut ",")

    Retourne:
    --------
    pd.DataFrame
        DataFrame contenant les données du fichier
    Exemple:
    --------
    df = get_cloud_csv("dvf")
    df = get_cloud_csv("dossier_complet", sep=";")
    """
    if not filename.endswith(".csv"):
        filename = f"{filename}.csv"

    S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
    fs = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"endpoint_url": S3_ENDPOINT_URL}
    )

    with fs.open(f"ylarre/{filename}", "rb") as f:
        df = pd.read_csv(
            f,
            sep=sep,
            dtype={"code_commune": "str"}
        )

    return df


def get_local_csv(filename, sep=','):
    """
    Charge un fichier CSV du dossier Données et retourne un DataFrame.
    Paramètres:
    -----------
    filename : str
        Nom du fichier CSV (avec ou sans l'extension .csv)
    Retourne:
    --------
    pd.DataFrame
        DataFrame contenant les données du CSV
    """
    # Chemin vers le dossier Données (relatif au dossier scripts)
    donnees_path = os.path.join(os.path.dirname(__file__), '..', 'Données')
    # Ajouter .csv si nécessaire
    if not filename.endswith('.csv'):
        filename += '.csv'
    # Chemin complet du fichier
    file_path = os.path.join(donnees_path, filename)
    # Charger et retourner le DataFrame
    return pd.read_csv(file_path, sep=sep)
