from flask import Flask, request
from telegram import Bot, Update
from telegram.constants import ParseMode
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import asyncio

# --------------------------
# CONFIGURACIÓN
# --------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
HISTORIAL_FILE = "historial.json"
MAX_HISTORIAL = 3           # máximo de mensajes guardados por usuario
TEMPERATURE = 0.7
MAX_RESPONSE_LENGTH = 40    # tokens de respuesta
MAX_PROMPT_LENGTH = 100     # tokens de prompt

bot = Bot(token=TELEGRAM_TOKEN)
app = Flask(__name__)

# --------------------------
# CARGAR MODELO MUY LIGERO
# --------------------------
print("Cargando modelo tiny-gpt2...")
tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
device = "cpu"
model.to(device)
model.eval()
print("Modelo cargado en CPU")

# --------------------------
# FUNCIONES DE HISTORIAL
# --------------------------
def cargar_historial():
    try:
        with open(HISTORIAL_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def guardar_historial(historial):
    with open(HISTORIAL_FILE, "w") as f:
        json.dump(historial, f)

def agregar_mensaje(user_id, rol, mensaje):
    historial = cargar_historial()
    if str(user_id) not in historial:
        historial[str(user_id)] = [{"role": "system", "content": "Eres un asistente amigable."}]
    historial[str(user_id)].append({"role": rol, "content": mensaje})
    if len(historial[str(user_id)]) > MAX_HISTORIAL + 1:
        historial[str(user_id)] = [historial[str(user_id)][0]] + historial[str(user_id)][-MAX_HISTORIAL:]
    guardar_historial(historial)

def obtener_historial(user_id):
    historial = cargar_historial()
    return historial.get(str(user_id), [])

def resetear_historial(user_id):
    historial = cargar_historial()
    historial[str(user_id)] = [{"role": "system", "content": "Eres un asistente amigable."}]
    guardar_historial(historial)

# --------------------------
# GENERAR RESPUESTA
# --------------------------
def gpt_responder(user_id, mensaje):
    agregar_mensaje(user_id, "user", mensaje)
    historial_usuario = obtener_historial(user_id)

    prompt = ""
    for m in historial_usuario:
        if m["role"] != "system":
            prompt += f"{m['role']}: {m['content']}\n"
    prompt += "assistant:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LENGTH).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs.input_ids.shape[1] + MAX_RESPONSE_LENGTH,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=TEMPERATURE
        )

    texto_respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    respuesta_final = texto_respuesta[len(prompt):].strip()
    agregar_mensaje(user_id, "assistant", respuesta_final)
    return respuesta_final or "🤖 No entendí eso, prueba otra vez."

# --------------------------
# ENVIAR MENSAJE
# --------------------------
def send_message(chat_id, text):
    try:
        asyncio.run(bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML))
    except Exception as e:
        print(f"Error enviando mensaje: {e}")

# --------------------------
# WEBHOOK
# --------------------------
@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        update = Update.de_json(request.get_json(force=True), bot)
        print("Update recibido:", update.to_dict())

        if update.message and update.message.text:
            user_id = update.message.from_user.id
            mensaje_usuario = update.message.text.strip()
            print(f"Mensaje de {user_id}: {mensaje_usuario}")

            if mensaje_usuario.lower() == "/reset":
                resetear_historial(user_id)
                send_message(update.message.chat.id, "✅ Historial reiniciado.")
            else:
                respuesta = gpt_responder(user_id, mensaje_usuario)
                send_message(update.message.chat.id, respuesta)
        else:
            print("Update sin texto.")
    except Exception as e:
        print(f"Error en webhook: {e}")
    return "ok"

# --------------------------
# ENDPOINTS PARA TEST / RAÍZ
# --------------------------
@app.route("/", methods=["GET"])
def root():
    return "Bot de Telegram activo ✅"

@app.route("/test", methods=["GET"])
def test():
    return "ok from Flask + tiny-gpt2!"

# --------------------------
# INICIAR SERVIDOR
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Servidor Flask ejecutándose en puerto {port}...")
    app.run(host="0.0.0.0", port=port)
