# Gradio Extra Trees Classifier Predictor

Este proyecto implementa una interfaz de usuario utilizando Gradio para predecir la probabilidad de reingreso de un paciente en menos de 30 días utilizando un modelo de Extra Trees Classifier.

## Estructura del Proyecto

```
gradio-et-predictor
├── src
│   ├── app.py          # Punto de entrada de la aplicación Gradio
│   └── utils.py        # Funciones auxiliares para el preprocesamiento de datos
├── artifacts
│   └── et_model.joblib # Modelo de Extra Trees Classifier entrenado
├── requirements.txt     # Dependencias necesarias para el proyecto
└── README.md            # Documentación del proyecto
```

## Instalación

1. Clona este repositorio en tu máquina local.
2. Navega al directorio del proyecto.
3. Instala las dependencias necesarias ejecutando:

```
pip install -r requirements.txt
```

## Uso

Para ejecutar la aplicación Gradio, utiliza el siguiente comando:

```
python src/app.py
```

Esto iniciará un servidor local y abrirá la interfaz de usuario en tu navegador.

## Ejemplo de Uso

Introduce las características del paciente en la interfaz y haz clic en el botón para predecir la probabilidad de reingreso en menos de 30 días. La aplicación mostrará el resultado en la pantalla.

## Contribuciones

Las contribuciones son bienvenidas. Si deseas contribuir, por favor abre un issue o envía un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT.