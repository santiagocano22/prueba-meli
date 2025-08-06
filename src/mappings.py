import pandas as pd

def load_mappings(ids_csv_path: str) -> dict[str, dict[int, str]]:
    """
    Lee el CSV de mapeos (tres secciones concatenadas) y devuelve
    un dict con tres sub-dicts:
      {
        'admission_type_id': {1: 'Emergency', …},
        'discharge_disposition_id': {1: 'Discharged to home', …},
        'admission_source_id': {1: 'Physician Referral', …}
      }
    """
    # Leer líneas no vacías
    with open(ids_csv_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    sections: dict[str, list[list[str]]] = {}
    current = None

    for line in lines:
        # Detectar cabeceras
        if line.startswith('admission_type_id'):
            current = 'admission_type_id'
            sections[current] = []
            continue
        if line.startswith('discharge_disposition_id'):
            current = 'discharge_disposition_id'
            sections[current] = []
            continue
        if line.startswith('admission_source_id'):
            current = 'admission_source_id'
            sections[current] = []
            continue
        # Todas las demás líneas, si estamos dentro de una sección
        if current:
            # Separar sólo en la primera coma
            key, desc = line.split(',', 1)
            sections[current].append((key, desc))

    # Convertir a dict[int,str]
    mappings: dict[str, dict[int, str]] = {}
    for section, pairs in sections.items():
        # Algunos IDs vienen como '6' o como texto 'NULL' → saltarlos o mapearlos a None
        d: dict[int, str] = {}
        for key, desc in pairs:
            try:
                d[int(key)] = desc.strip()
            except ValueError:
                # Si key no es entero (por ejemplo 'NULL'), lo ignoramos
                continue
        mappings[section] = d

    return mappings


def apply_mappings(df: pd.DataFrame, mappings: dict[str, dict[int, str]]) -> pd.DataFrame:
    """
    Añade al df tres columnas *_desc con su descripción:
      - admission_type_desc
      - discharge_disposition_desc
      - admission_source_desc
    """
    df = df.copy()

    # Asegurarnos de que las columnas sean enteros (o NaN si no se puede)
    for col in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Mapear y crear las columnas *_desc
    df['admission_type_desc'] = df['admission_type_id'].map(mappings['admission_type_id'])
    df['discharge_disposition_desc'] = df['discharge_disposition_id'].map(mappings['discharge_disposition_id'])
    df['admission_source_desc'] = df['admission_source_id'].map(mappings['admission_source_id'])

    return df
