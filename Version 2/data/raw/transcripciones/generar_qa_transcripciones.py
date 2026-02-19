import json
import os

INPUT_FOLDER = "./"
OUTPUT_FILE = "../../processed/transcripciones_qa.jsonl"

def dividir_en_bloques(texto, tamaño=1500):
    bloques = []
    for i in range(0, len(texto), tamaño):
        bloques.append(texto[i:i+tamaño])
    return bloques

def generar_par_qa(bloque):
    pregunta = "Explica el siguiente contenido sobre Redes Locales."
    respuesta = bloque.strip()
    return {
        "messages": [
            {"role": "user", "content": pregunta},
            {"role": "assistant", "content": respuesta}
        ]
    }

def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for archivo in os.listdir(INPUT_FOLDER):
            if archivo.endswith(".txt"):
                with open(archivo, "r", encoding="utf-8") as f:
                    texto = f.read()

                bloques = dividir_en_bloques(texto)

                for bloque in bloques:
                    ejemplo = generar_par_qa(bloque)
                    out.write(json.dumps(ejemplo, ensure_ascii=False) + "\n")

    print("Q/A de transcripciones generado correctamente.")

if __name__ == "__main__":
    main()

