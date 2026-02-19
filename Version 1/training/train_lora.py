#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entrenamiento LoRA para el Profesor Virtual de Redes Locales
Modelo base: Qwen2.5-0.5B-Instruct
Guarda adaptador LoRA + registra entrenamiento en MySQL
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# 👇 IMPORTANTE: conexión a MySQL
from database.db import registrar_entrenamiento


# ===============================
# CONFIGURACIÓN
# ===============================

MODEL_NAME = "sshleifer/tiny-gpt2"
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset_final.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "profesor_redes_lora")


MAX_LENGTH = 128
EPOCHS = 1
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 4


# ===============================
# MAIN
# ===============================

def main():

    print("🚀 Iniciando entrenamiento LoRA - Profesor Virtual Redes")
    print("Modelo base:", MODEL_NAME)
    print("Dataset:", DATASET_PATH)
    print("-" * 60)

    # -----------------------------------
    # 1️⃣ Cargar dataset
    # -----------------------------------

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("No se encontró el dataset_final.jsonl")

    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    print(f"📊 Ejemplos cargados: {len(dataset)}")

    # -----------------------------------
    # 2️⃣ Cargar tokenizer
    # -----------------------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------
    # 3️⃣ Cargar modelo base (4bit si hay GPU)
    # -----------------------------------

    if torch.cuda.is_available():
        print("💻 GPU detectada → Entrenamiento 4-bit")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            load_in_4bit=True,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
        use_fp16 = True
    else:
        print("💻 CPU detectada → Entrenamiento estándar")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        use_fp16 = False

        # -----------------------------------
    # 4️⃣ Configurar LoRA
    # -----------------------------------

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()



    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -----------------------------------
    # 5️⃣ Convertir messages → texto plano
    # -----------------------------------

    def formatear_chat(example):
        texto = ""
        for m in example["messages"]:
            if m["role"] == "user":
                texto += f"Usuario: {m['content']}\n"
            else:
                texto += f"Asistente: {m['content']}\n"
        return {"text": texto}

    dataset = dataset.map(formatear_chat, remove_columns=dataset.column_names)

    # -----------------------------------
    # 6️⃣ Tokenización
    # -----------------------------------

    def tokenizar(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = dataset.map(tokenizar, batched=True, remove_columns=["text"])

    # -----------------------------------
    # 7️⃣ Argumentos de entrenamiento
    # -----------------------------------

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        fp16=use_fp16,
        report_to="none"
    )

    # -----------------------------------
    # 8️⃣ Crear Trainer
    # -----------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    # -----------------------------------
    # 9️⃣ Entrenar
    # -----------------------------------

    print("🚂 Entrenando...")
    trainer.train()
    print("🏁 Entrenamiento finalizado")

    # -----------------------------------
    # 🔟 Guardar modelo LoRA
    # -----------------------------------

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"💾 Modelo guardado en: {OUTPUT_DIR}")

    # -----------------------------------
    # 1️⃣1️⃣ Registrar entrenamiento en MySQL
    # -----------------------------------

    registrar_entrenamiento(
        modelo="ProfesorRedes_v1",
        epochs=EPOCHS,
        dataset_size=len(dataset)
    )

    print("📊 Entrenamiento registrado en MySQL correctamente.")
    print("✅ Todo listo.")


if __name__ == "__main__":
    main()

