<!DOCTYPE html>
<html>
<head>
    <title>Profesor Virtual Redes</title>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>

<div class="chat-container">
    <h1>Profesor Virtual de Redes Locales</h1>

    <div id="chat"></div>

    <input type="text" id="pregunta" placeholder="Haz tu pregunta...">
    <button onclick="enviar()">Enviar</button>
</div>

<script>
async function enviar() {
    const pregunta = document.getElementById("pregunta").value;

    const response = await fetch("/preguntar", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({pregunta})
    });

    const data = await response.json();

    const chat = document.getElementById("chat");
    chat.innerHTML += "<p><strong>Tú:</strong> " + pregunta + "</p>";
    chat.innerHTML += "<p><strong>Profesor:</strong> " + data.respuesta + "</p>";

    document.getElementById("pregunta").value = "";
}
</script>

</body>
</html>

