import mysql.connector

DB_CONFIG = {
    "host": "localhost",
    "user": "root",        # cambia si tu usuario es distinto
    "password": "",        # pon tu contraseña si tienes
    "database": "profesor_virtual_redes"
}

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

def guardar_conversacion(pregunta, respuesta, modelo):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        INSERT INTO conversaciones (pregunta, respuesta, modelo_usado)
        VALUES (%s, %s, %s)
    """
    cursor.execute(query, (pregunta, respuesta, modelo))

    conn.commit()
    cursor.close()
    conn.close()

def registrar_entrenamiento(modelo, epochs, dataset_size):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        INSERT INTO entrenamientos (modelo, epochs, dataset_size)
        VALUES (%s, %s, %s)
    """
    cursor.execute(query, (modelo, epochs, dataset_size))

    conn.commit()
    cursor.close()
    conn.close()

