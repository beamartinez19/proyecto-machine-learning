import pandas as pd



def data_report(df):
    # Sacamos los NOMBRES
    cols = pd.DataFrame(df.columns.values, columns=["COL_N"])

    # Sacamos los TIPOS
    types = pd.DataFrame(df.dtypes.values, columns=["DATA_TYPE"])

    # Sacamos los MISSINGS
    percent_missing = round(df.isnull().sum() * 100 / len(df), 2)
    percent_missing_df = pd.DataFrame(percent_missing.values, columns=["MISSINGS (%)"])

    # Sacamos los VALORES UNICOS
    unicos = pd.DataFrame(df.nunique().values, columns=["UNIQUE_VALUES"])
    
    percent_cardin = round(unicos['UNIQUE_VALUES']*100/len(df), 2)
    percent_cardin_df = pd.DataFrame(percent_cardin.values, columns=["CARDIN (%)"])

    concatenado = pd.concat([cols, types, percent_missing_df, unicos, percent_cardin_df], axis=1, sort=False)
    concatenado.set_index('COL_N', drop=True, inplace=True)


    return concatenado.T

def cumplimentar_missing_usuario(df, group_col):
    """
    Esta función cumplimenta los missings de cada fila que hay en todas las columnas con los valores correspondientes de otras filas
    del mismo usuario, agrupando por la columna especificada en group_col.
    
    """
    # Agrupo por ID del empleado y utilizo fillna para rellenar los valores faltantes en cada grupo
    df_filled = df.groupby(group_col).fillna(method='ffill').fillna(method='bfill')
    
    return df_filled
