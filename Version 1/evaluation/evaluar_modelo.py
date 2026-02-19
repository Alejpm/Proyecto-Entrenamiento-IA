import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

MODEL_PATH = "../models/profesor_redes_final"
TEST_DATA = "test_dataset.jsonl"

def generar_respuesta(model, tokenizer, pregunta):
    inputs = tokenizer(pregunta, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():

    dataset = load_dataset("json", data_files=TEST_DATA, split="train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

    score = 0
    total = len(dataset)

    for example in dataset:
        pregunta = example["question"]
        respuesta_real = example["answer"]

        respuesta_modelo = generar_respuesta(model, tokenizer, pregunta)

        if respuesta_real.lower()[:30] in respuesta_modelo.lower():
            score += 1

    print("Precisión aproximada:", score / total)

if __name__ == "__main__":
    main()

