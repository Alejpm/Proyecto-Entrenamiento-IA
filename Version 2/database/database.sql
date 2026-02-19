CREATE DATABASE profesor_virtual_redes CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE TABLE conversaciones (
    id INT AUTO_INCREMENT PRIMARY KEY,
    pregunta TEXT NOT NULL,
    respuesta TEXT NOT NULL,
    modelo_usado VARCHAR(100),
    fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE entrenamientos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    modelo VARCHAR(100),
    epochs INT,
    dataset_size INT,
    fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

