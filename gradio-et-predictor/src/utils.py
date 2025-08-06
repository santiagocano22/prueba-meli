def preprocess_input(input_data):
    # Aquí puedes agregar la lógica para preprocesar los datos de entrada
    # Asegúrate de que el formato de input_data sea compatible con el modelo
    return input_data

def load_model(model_path):
    import joblib
    model = joblib.load(model_path)
    return model

def predict(model, input_features):
    # Realiza la predicción utilizando el modelo cargado
    prediction = model.predict_proba(input_features)[:, 1]  # Probabilidad de la clase positiva
    return prediction