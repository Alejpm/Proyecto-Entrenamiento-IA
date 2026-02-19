import os
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from database.db import guardar_conversacion

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "profesor_redes_final")

print("📦 Cargando modelo desde:", MODEL_PATH)

# Cargar tokenizer y modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# 🔥 MUY IMPORTANTE PARA GPT2
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/preguntar", methods=["POST"])
def preguntar():
    try:
        data = request.get_json()
        pregunta = data.get("pregunta", "").strip()

        if not pregunta:
            return jsonify({"respuesta": "No has escrito ninguna pregunta."})

        prompt = f"Pregunta: {pregunta}\nRespuesta:"

        inputs = tokenizer(prompt, return_tensors="pt")

        if inputs["input_ids"].shape[1] == 0:
            return jsonify({"respuesta": "No se pudo procesar la pregunta."})

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=60,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        if outputs.shape[0] == 0:
            return jsonify({"respuesta": "El modelo no generó respuesta."})

        texto_generado = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Respuesta:" in texto_generado:
            respuesta = texto_generado.split("Respuesta:")[-1].strip()
        else:
            respuesta = texto_generado

        if not respuesta:
            respuesta = "El modelo no pudo generar una respuesta coherente."

        guardar_conversacion(pregunta, respuesta, "ProfesorRedes_v1")

        return jsonify({"respuesta": respuesta})

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"respuesta": "Error interno del servidor."})

if __name__ == "__main__":
    app.run(debug=True)

