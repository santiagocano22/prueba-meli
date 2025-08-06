import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def overview(df: pd.DataFrame) -> None:
    """Shape, tipos y primeros estadísticos generales."""
    print("■ Shape:", df.shape)
    print("■ Columnas y tipos:\n", df.dtypes)
    print("\n■ Estadísticos numéricos:\n", df.describe(include='number'))
    print("\n■ Estadísticos categóricos:\n", df.describe(include='object'))

def plot_categorical(df: pd.DataFrame, cols: list[str], figsize=(6,4)) -> None:
    """Bar plots de cada columna categórica en cols."""
    for col in cols:
        plt.figure(figsize=figsize)
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f"Distribución de {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def plot_numeric(df: pd.DataFrame, cols: list[str], bins=30, figsize=(6,4)) -> None:
    """Histogramas de las columnas numéricas en cols."""
    for col in cols:
        plt.figure(figsize=figsize)
        df[col].astype(float).hist(bins=bins)
        plt.title(f"{col} (histograma)")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.show()

def correlation_matrix(df: pd.DataFrame, cols: list[str], figsize=(8,6)) -> None:
    """Mapa de calor de correlación entre cols numéricas."""
    corr = df[cols].astype(float).corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Matriz de correlación")
    plt.show()

def top_categories(df: pd.DataFrame, col: str, top_n: int = 10) -> pd.DataFrame:
    """Devuelve las top_n categorías con count y pct."""
    vc = df[col].value_counts().head(top_n)
    pct = vc / len(df) * 100
    return pd.DataFrame({'count': vc, 'pct': pct})


def med_change_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    summary = {}
    for col in cols:
        vc = df[col].value_counts(normalize=True) * 100
        summary[col] = vc
    return pd.DataFrame(summary).fillna(0)


def cat_target_crosstab(
    df: pd.DataFrame,
    cat_cols: List[str],
    target: str
) -> Dict[str, pd.DataFrame]:
    """
    Para cada columna en cat_cols devuelve un crosstab (% por fila):
      índice = categorías de la variable,
      columnas = clases de target,
      valores = porcentaje de cada clase dentro de la categoría.
    """
    total = len(df)
    result = {}
    for col in cat_cols:
        ct = pd.crosstab(df[col], df[target], normalize='index') * 100
        ct = ct.round(2)
        result[col] = ct
    return result

def plot_cat_target_relation(
    df: pd.DataFrame,
    col: str,
    target: str,
    figsize=(6,4)
) -> None:
    """
    Grafica un barplot apilado (proporción) de target para cada categoría de col.
    """
    ct = pd.crosstab(df[col], df[target], normalize='index')
    ct.plot(kind='bar', stacked=True, figsize=figsize)
    plt.title(f"Distribución de '{target}' por '{col}'")
    plt.ylabel("Proporción")
    plt.xticks(rotation=45)
    plt.legend(title=target, bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

def value_counts_report(
    df: pd.DataFrame,
    cols: List[str],
    top_n: int = 10,
    dropna: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Para cada columna en `cols`, calcula:
      - count: conteo de cada valor (incluyendo NaN si dropna=False)
      - pct  : porcentaje sobre el total de filas
    Devuelve un dict: columna -> DataFrame con columnas ['count','pct'].
    """
    total = len(df)
    reports: Dict[str, pd.DataFrame] = {}

    for col in cols:
        vc = df[col].value_counts(dropna=dropna).rename("count")
        pct = (vc / total * 100).rename("pct")
        rpt = pd.concat([vc, pct], axis=1).reset_index().rename(
            columns={"index": col}
        )
        reports[col] = rpt.head(top_n)

    return reports

def count_missing_and_sentinels(df: pd.DataFrame,
                                sentinels: list[str] = None
                               ) -> pd.Series:
    """
    Para cada columna de df, cuenta cuántos registros son:
      - nulos (pd.isna)
      - cadenas en sentinels ("" , "?", "NULL", "NaN")
    Devuelve una Serie ordenada descendente con el conteo,
    y solo incluye columnas con al menos 1 incidencia.
    """
    if sentinels is None:
        sentinels = ['', '?', 'NULL', 'NaN', 'abcde']

    counts: dict[str,int] = {}
    for col in df.columns:
        s = df[col]
        # máscara de verdaderos nulos
        mask_null = s.isna()
        # convertir todo a str y strip, luego chequear sentinels
        mask_sentinel = s.astype(str).str.strip().isin(sentinels)
        # unión de ambos
        mask = mask_null | mask_sentinel
        total = int(mask.sum())
        if total > 0:
            counts[col] = total

    # convertir a Serie y ordenar
    return pd.Series(counts, name='missing_count') \
             .sort_values(ascending=False)
