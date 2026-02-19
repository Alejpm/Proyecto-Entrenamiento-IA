import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_MODEL = "sshleifer/tiny-gpt2"
LORA_DIR = os.path.join(BASE_DIR, "models", "profesor_redes_lora")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "profesor_redes_final")

def main():
    print("🔄 Iniciando merge del modelo...")

    # Cargar modelo base
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # Cargar LoRA
    model = PeftModel.from_pretrained(base_model, LORA_DIR)

    # Merge
    model = model.merge_and_unload()

    # Guardar modelo final
    model.save_pretrained(OUTPUT_DIR)

    # 🔥 MUY IMPORTANTE: guardar tokenizer del modelo base
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ Modelo final guardado en:", OUTPUT_DIR)

if __name__ == "__main__":
    main()

