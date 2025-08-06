import pandas as pd
import numpy as np
from typing import List, Dict, Union
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------
DEFAULT_SENTINELS = ['', '?', 'NULL', 'NaN', 'abcde']

# --------------------------------------------------------------------------------
# Missing-data inspection & pipeline
# --------------------------------------------------------------------------------
def get_missing_stats(
    df: pd.DataFrame,
    sentinels: List[str] = DEFAULT_SENTINELS
) -> pd.DataFrame:
    """
    Devuelve un DataFrame con, para cada columna:
      - n_missing: count de NaN reales + sentinels
      - pct_missing: porcentaje sobre el total de filas
    """
    total = len(df)
    stats = []
    for col in df.columns:
        ser = df[col]
        mask_null = ser.isna()
        mask_sentinel = ser.astype(str).str.strip().isin(sentinels)
        n_miss = int((mask_null | mask_sentinel).sum())
        stats.append({
            'column': col,
            'n_missing': n_miss,
            'pct_missing': n_miss / total * 100
        })
    return (pd.DataFrame(stats)
            .query("n_missing > 0")
            .sort_values('pct_missing', ascending=False)
            .reset_index(drop=True))


def drop_columns_by_missing(
    df: pd.DataFrame,
    pct_threshold: float = 90.0,
    sentinels: List[str] = DEFAULT_SENTINELS
) -> pd.DataFrame:
    """
    Elimina columnas que tengan >= pct_threshold% de missing
    (NaN o sentinels).
    """
    stats = get_missing_stats(df, sentinels)
    to_drop = stats.loc[stats.pct_missing >= pct_threshold, 'column'].tolist()
    return df.drop(columns=to_drop, errors='ignore')


def replace_sentinels_with_nan(
    df: pd.DataFrame,
    sentinels: List[str] = DEFAULT_SENTINELS
) -> pd.DataFrame:
    """
    Reemplaza en TODO el DataFrame los valores sentinel por np.nan.
    Sólo modifica columnas de tipo object; deja intactos otros tipos.
    """
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].replace(sentinels, np.nan)
    return df


def impute_missing(
    df: pd.DataFrame,
    numeric_strategy: str = 'median',
    categorical_strategy: str = 'constant',
    fill_value: Union[str, float] = 'Unknown'
) -> pd.DataFrame:
    """
    Imputa missing en:
      - columnas numéricas: con SimpleImputer(strategy=numeric_strategy)
      - columnas categóricas: con SimpleImputer(strategy=categorical_strategy, fill_value)
    Devuelve un DataFrame con las columnas imputadas y el resto intacto.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if num_cols:
        num_imp = SimpleImputer(strategy=numeric_strategy)
        df[num_cols] = num_imp.fit_transform(df[num_cols])

    if cat_cols:
        cat_imp = SimpleImputer(strategy=categorical_strategy, fill_value=fill_value)
        df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    return df


def impute_specific_categorical(
    df: pd.DataFrame,
    fill_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Imputa columnas categóricas con un valor específico por columna.
    
    fill_map: { columna: valor_a_usar_en_fillna }
    """
    df = df.copy()
    for col, val in fill_map.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df


def clean_missing_pipeline(
    df: pd.DataFrame,
    drop_pct: float = 90.0,
    sentinels: List[str] = DEFAULT_SENTINELS,
    numeric_strategy: str = 'median',
    categorical_strategy: str = 'constant',
    fill_value: Union[str, float] = 'Unknown'
) -> pd.DataFrame:
    """
    Pipeline todo-en-uno:
      1. Drop cols con ≥drop_pct% missing
      2. Reemplaza sentinels por NaN
      3. Imputa numéricos y categóricos según estrategias dadas
    """
    return (
        df
        .pipe(drop_columns_by_missing, pct_threshold=drop_pct, sentinels=sentinels)
        .pipe(replace_sentinels_with_nan, sentinels=sentinels)
        .pipe(impute_missing,
              numeric_strategy=numeric_strategy,
              categorical_strategy=categorical_strategy,
              fill_value=fill_value)
    )

# --------------------------------------------------------------------------------
# General Feature Engineering
# --------------------------------------------------------------------------------

def drop_columns(
    df: pd.DataFrame,
    cols: List[str]
) -> pd.DataFrame:
    """Elimina columnas no deseadas."""
    return df.drop(columns=cols, errors='ignore')


def truncate_icd9(
    df: pd.DataFrame,
    cols: List[str]
) -> pd.DataFrame:
    """
    Deja solo los primeros 3 dígitos de cada código ICD9.
    """
    df = df.copy()
    for col in cols:
        df[col] = df[col].str[:3]
    return df


def group_rare_categories(
    df: pd.DataFrame,
    col: str,
    top_n: int = 20,
    other_label: str = 'Other'
) -> pd.DataFrame:
    """
    En la columna col deja solo las top_n categorías más frecuentes,
    el resto las marca como other_label.
    """
    df = df.copy()
    top = df[col].value_counts().nlargest(top_n).index
    df[col] = df[col].where(df[col].isin(top), other_label)
    return df


def med_change_counts(
    df: pd.DataFrame,
    med_cols: List[str],
    statuses: List[str] = ['up','down','steady','no']
) -> pd.DataFrame:
    """
    Por cada status crea una columna n_med_{status}
    con el número de fármacos que presentan ese cambio.
    """
    df = df.copy()
    for status in statuses:
        df[f'n_med_{status}'] = (df[med_cols] == status).sum(axis=1)
    return df


def create_age_numeric(
    df: pd.DataFrame,
    age_col: str = 'age'
) -> pd.DataFrame:
    """
    Convierte rangos de edad ('[30-40)') a edad numérica promedio.
    """
    df = df.copy()
    df['age_num'] = (
        df[age_col]
          .str.replace(r'[\[\)\+]', '', regex=True)
          .str.split('-')
          .apply(lambda x: (int(x[0]) + int(x[1])) / 2)
    )
    return df


def bin_numeric(
    df: pd.DataFrame,
    col: str,
    bins: List[float],
    labels: List[str] = None
) -> pd.DataFrame:
    """
    Crea una nueva columna '{col}_bin' categorizando valores según bins y labels.
    """
    df = df.copy()
    df[f'{col}_bin'] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    return df


def one_hot_encode(
    df: pd.DataFrame,
    cols: List[str]
) -> pd.DataFrame:
    """
    One-hot encode de las columnas en cols.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    arr = encoder.fit_transform(df[cols])
    df_ohe = pd.DataFrame(
        arr,
        columns=encoder.get_feature_names_out(cols),
        index=df.index
    )
    return pd.concat([df.drop(columns=cols), df_ohe], axis=1)


def ordinal_encode(
    df: pd.DataFrame,
    cols: List[str],
    ordering: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Aplica OrdinalEncoder con orden específico por cada col en cols.
    ordering: {col: [cat1, cat2, ...]}
    """
    df = df.copy()
    for col in cols:
        enc = OrdinalEncoder(categories=[ordering[col]])
        df[[col]] = enc.fit_transform(df[[col]])
    return df


def scale_numeric(
    df: pd.DataFrame,
    cols: List[str]
) -> pd.DataFrame:
    """
    Aplica StandardScaler sobre cols y reemplaza sus valores.
    """
    df = df.copy()
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def add_interaction(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    new_col: str
) -> pd.DataFrame:
    """
    Crea una nueva columna new_col = df[col_a] * df[col_b].
    """
    df = df.copy()
    df[new_col] = df[col_a] * df[col_b]
    return df
