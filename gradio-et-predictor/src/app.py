from sklearn.ensemble import ExtraTreesClassifier
import joblib
import gradio as gr
import pandas as pd

# Ruta y carga del modelo
model_path = 'artifacts/et_model.joblib'
model = joblib.load(model_path)

# Lista de features seleccionadas
FEATURES = [
    'number_inpatient', 'number_diagnoses', 'time_in_hospital', 'A1Cresult', 'change', 'num_lab_procedures', 'num_medications',
    'number_emergency', 'num_procedures', 'insulin', 'diabetesMed', 'number_outpatient', 'metformin', 'age_num',
    'diag_1_250', 'diag_1_414', 'diag_1_428', 'diag_1_434', 'diag_1_786', 'diag_2_401', 'diag_2_403', 'diag_2_414',
    'payer_code_MC', 'payer_code_Unknown', 'medical_specialty_Cardiology', 'medical_specialty_ObstetricsandGynecology',
    'medical_specialty_Unknown', 'diag_3_272', 'diag_3_401', 'diag_3_403', 'diag_3_585', 'diag_3_Unk',
    'discharge_disposition_desc_Discharged to home', 'discharge_disposition_desc_Discharged/transferred to SNF',
    'discharge_disposition_desc_Discharged/transferred to another rehab fac including rehab units of a hospital .',
    'discharge_disposition_desc_Discharged/transferred to another short term hospital',
    'discharge_disposition_desc_Discharged/transferred to another type of inpatient care institution',
    'discharge_disposition_desc_Discharged/transferred to home with home health service',
    'discharge_disposition_desc_Discharged/transferred within this institution to Medicare approved swing bed',
    'discharge_disposition_desc_Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital',
    'discharge_disposition_desc_Expired', 'admission_source_desc_Emergency Room'
]

# Definir la función de predicción
def predict_reingreso(*inputs):
    input_dict = dict(zip(FEATURES, inputs))
    X = pd.DataFrame([input_dict])
    prob = model.predict_proba(X)[0, 1]
    return f"Probabilidad de reingreso <30 días: {prob:.2%}"

# Crear inputs de Gradio (todos como Number, permiten negativos)
inputs = [gr.Number(label=feat, precision=3) for feat in FEATURES]

# Crear interfaz
gr.Interface(
    fn=predict_reingreso,
    inputs=inputs,
    outputs=gr.Textbox(label="Resultado"),
    title="Predicción de Reingreso (<30 días)",
    description="Introduce los valores de las variables seleccionadas para obtener la probabilidad de reingreso en menos de 30 días."
).launch()