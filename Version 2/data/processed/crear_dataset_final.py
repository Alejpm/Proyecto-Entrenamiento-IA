import os

BASE_DIR = os.path.dirname(__file__)

INPUT_FILES = [
    os.path.join(BASE_DIR, "pdf_qa.jsonl"),
    os.path.join(BASE_DIR, "transcripciones_qa.jsonl")
]

OUTPUT_FILE = os.path.join(BASE_DIR, "dataset_final.jsonl")

def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        total = 0
        for file in INPUT_FILES:
            if os.path.exists(file):
                with open(file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            out.write(line)
                            total += 1

    print(f"Dataset final generado correctamente. Ejemplos: {total}")

if __name__ == "__main__":
    main()

