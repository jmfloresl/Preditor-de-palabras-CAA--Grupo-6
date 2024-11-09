from flask import Flask, render_template, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import re
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import words

app = Flask(__name__)

# Descargar el corpus de palabras en español
nltk.download('cess_esp')
spanish_words = set(w.lower() for w in nltk.corpus.cess_esp.words())

# Cargar modelo pre-entrenado de lenguaje en español
model_name = "dccuchile/bert-base-spanish-wwm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, is_decoder=True)

# Inicializar el pipeline de completado de texto
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data.get('input', '').strip()

    # Generar predicciones
    predictions = generate_predictions(user_input)

    return jsonify({'predictions': predictions})

def generate_predictions(user_input):
    predictions = set()
    
    # Generar predicciones basadas en coincidencia de patrones
    pattern_predictions = generate_pattern_predictions(user_input)
    predictions.update(pattern_predictions)
    
    # Generar predicciones basadas en el modelo de lenguaje
    model_predictions = generate_model_predictions(user_input)
    predictions.update(model_predictions)
    
    # Generar predicciones basadas en coincidencia difusa
    fuzzy_predictions = generate_fuzzy_predictions(user_input)
    predictions.update(fuzzy_predictions)
    
    # Ordenar y filtrar predicciones
    sorted_predictions = sorted(predictions, key=lambda x: -relevance_score(x, user_input))
    
    # Si no hay predicciones, devolver una lista con el input del usuario
    if not sorted_predictions:
        return [user_input]
    
    return sorted_predictions[:10]  # Devolver las 10 mejores predicciones

def generate_pattern_predictions(input_text):
    common_completions = {
        'com': ['como', 'comer', 'comida', 'comenzar', 'compartir'],
        'est': ['esta', 'estar', 'estoy', 'estamos', 'estás'],
        'hol': ['hola', 'hola como estás', 'hola buenos días', 'hola a todos'],
        'buen': ['bueno', 'buenos días', 'buenas tardes', 'buenas noches'],
        'ayud': ['ayuda', 'ayudar', 'ayudame', 'ayudante'],
        'medi': ['medicina', 'medico', 'medica', 'medicamento'],
        'past': ['pastilla', 'pasta', 'pastel', 'pastor'],
        'nece':['necesito','necesito comer','necesito mis medicamentos','necesito ir al baño','necesito bañarme' ],
        'llam':['llamar', 'llamame','llama a mis padres','llama a mis hijos','llama al medico'],
        'ir':['ir al parque','ir al medico','ir a dormir','ir a la escuela','ir de viaje','ir con mis padres','ir con mis hijos'],
        # si necesitamos añadir más patrones comunes los escribimos aquí
    }
    
    predictions = []
    for key, completions in common_completions.items():
        if input_text.lower().startswith(key):
            predictions.extend(completions)
    
    return predictions

def generate_model_predictions(input_text):
    try:
        generated = text_generator(input_text, max_length=len(input_text) + 20, num_return_sequences=5, do_sample=True, temperature=0.7)
        
        predictions = []
        for item in generated:
            prediction_text = item['generated_text'][len(input_text):].strip()
            if prediction_text:
                words = prediction_text.split()
                if words:
                    prediction = words[0]
                    if prediction.lower() in spanish_words:
                        predictions.append(input_text + prediction)
        
        return predictions
    except Exception as e:
        print(f"Error en generate_model_predictions: {e}")
        return []

def generate_fuzzy_predictions(input_text):
    predictions = []
    for word in spanish_words:
        if fuzz.partial_ratio(input_text.lower(), word) > 80:  # Ajusta este umbral según sea necesario
            predictions.append(word)
    
    return predictions[:5]  # Limitar a las 5 mejores coincidencias difusas

def relevance_score(prediction, user_input):
    # Calcula un puntaje de relevancia basado en la coincidencia con el input del usuario
    if prediction.lower().startswith(user_input.lower()):
        return len(user_input) + len(prediction)
    return fuzz.partial_ratio(user_input.lower(), prediction.lower())

if __name__ == '__main__':
    app.run(debug=True)