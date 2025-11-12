from flask import Flask, request
from telegram import Bot, Update
from telegram.constants import ParseMode
import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

# --------------------------
# CONFIGURACIÓN
# --------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
HISTORIAL_FILE = "historial.json"
MAX_HISTORIAL = 5
TEMPERATURE = 0.7
MAX_RESPONSE_LENGTH = 100
MAX_PROMPT_LENGTH = 256

bot = Bot(token=TELEGRAM_TOKEN)
app = Flask(__name__)

# --------------------------
# CARGAR MODELO TinyLlama
# --------------------------
print("Cargando modelo TinyLlama-Chat...")
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print("Modelo cargado correctamente en", device)

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
        historial[str(user_id)] = [{"role": "system", "content": "Eres un asistente útil y amable."}]
    historial[str(user_id)].append({"role": rol, "content": mensaje})
    if len(historial[str(user_id)]) > MAX_HISTORIAL + 1:
        historial[str(user_id)] = [historial[str(user_id)][0]] + historial[str(user_id)][-MAX_HISTORIAL:]
    guardar_historial(historial)

def obtener_historial(user_id):
    historial = cargar_historial()
    return historial.get(str(user_id), [])

def resetear_historial(user_id):
    historial = cargar_historial()
    historial[str(user_id)] = [{"role": "system", "content": "Eres un asistente útil y amable."}]
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
            temperature=TEMPERATURE,
            top_p=0.9
        )

    texto_respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    respuesta_final = texto_respuesta[len(prompt):].strip()
    agregar_mensaje(user_id, "assistant", respuesta_final)
    return respuesta_final or "🤖 No he entendido eso, ¿puedes repetirlo?"

# --------------------------
# ENVÍO SINCRÓNICO A TELEGRAM
# --------------------------
def send_message_sync(chat_id, text):
    try:
        asyncio.run(bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.HTML))
    except Exception as e:
        print(f"Error enviando mensaje a Telegram: {e}")

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
            print(f"Mensaje recibido de {user_id}: {mensaje_usuario}")

            if mensaje_usuario.lower() == "/reset":
                resetear_historial(user_id)
                send_message_sync(update.message.chat.id, "✅ Historial reiniciado.")
            else:
                respuesta = gpt_responder(user_id, mensaje_usuario)
                send_message_sync(update.message.chat.id, respuesta)
        else:
            print("Update sin texto.")
    except Exception as e:
        print(f"Error en webhook: {e}")
    return "ok"

# --------------------------
# TEST ENDPOINT
# --------------------------
@app.route("/test", methods=["GET"])
def test():
    return "ok from Flask + TinyLlama!"

# --------------------------
# INICIAR SERVIDOR
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Servidor Flask ejecutándose en puerto {port}...")
    app.run(host="0.0.0.0", port=port)
