import os
import json
import pdfplumber

INPUT_FOLDER = "./"
OUTPUT_FILE = "../../processed/pdf_qa.jsonl"

def extraer_texto_pdf(ruta):
    texto = ""
    with pdfplumber.open(ruta) as pdf:
        for pagina in pdf.pages:
            texto += pagina.extract_text() + "\n"
    return texto

def dividir_en_bloques(texto, tamaño=1500):
    return [texto[i:i+tamaño] for i in range(0, len(texto), tamaño)]

def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for archivo in os.listdir(INPUT_FOLDER):
            if archivo.endswith(".pdf"):
                texto = extraer_texto_pdf(archivo)
                bloques = dividir_en_bloques(texto)

                for bloque in bloques:
                    ejemplo = {
                        "messages": [
                            {"role": "user", "content": "Explica el siguiente contenido técnico sobre Redes Locales."},
                            {"role": "assistant", "content": bloque.strip()}
                        ]
                    }
                    out.write(json.dumps(ejemplo, ensure_ascii=False) + "\n")

    print("Q/A de PDFs generado correctamente.")

if __name__ == "__main__":
    main()

