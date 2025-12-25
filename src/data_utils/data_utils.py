import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def read_data(path):
    df = pd.read_excel(path)
    return df


def prepare_dummy_cols(df, dummy_cols):

    df_dummy = pd.get_dummies(df, columns=dummy_cols, drop_first=True)

    del df_dummy['Contador']

    df_dummy["Pago"] = df_dummy["Pago"].map({"Anual":1, "Semestral":0})

    df_dummy["Domiciliacion"] = df_dummy["Domiciliacion"].map({"SI":1, "NO":0})

    df_dummy["SegundoConductor"] = df_dummy["SegundoConductor"].map({"SI":1, "NO":0})

    num_vars = ['Anyomatricula', 'Prima', 'ValorVehículo', 'Socioec', 'Antigüedad', 'Carnet']

    # print(df_dummy.head())
    # print(df_dummy.describe())
    #
    # print(df_dummy.columns)

    # feature_cols = ['Pago', 'Domiciliacion', 'Anyomatricula', 'Prima', 'Valor',
    #    'ValorVehículo', 'Motor', 'Canal', 'Socioec', 'Antigüedad', 'Edad',
    #    'Carnet', 'SegundoConductor', 'Figuras', 'Tipo_Furgoneta',
    #    'Tipo_Moto', 'Tipo_Turismo', 'Zonas_Zona1', 'Zonas_Zona2',
    #    'Zonas_Zona3', 'Zonas_Zona4', 'Zonas_Zona5', 'Zonas_Zona6',
    #    'Zonas_Zona7', 'Zonas_Zona8']

    feature_cols = ['Pago', 'Domiciliacion', 'Anyomatricula', 'Prima', 'Valor',
                    'ValorVehículo', 'Motor', 'Canal', 'Socioec', 'Antigüedad', 'Carnet',
                    'SegundoConductor', 'Figuras', 'Tipo_Furgoneta',
                    'Tipo_Moto', 'Tipo_Turismo', 'Zonas_Zona1', 'Zonas_Zona2',
                    'Zonas_Zona3', 'Zonas_Zona4', 'Zonas_Zona5', 'Zonas_Zona6',
                    'Zonas_Zona7', 'Zonas_Zona8']
    return df_dummy


def normalize_num_vars(df, num_vars):

    scaler = StandardScaler()
    X_scaled = df
    X_scaled[num_vars] = scaler.fit_transform(df[num_vars])

    return X_scaled