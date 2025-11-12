from flask import Flask, request
from telegram import Bot, Update
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

# --------------------------
# CONFIGURACIÓN
# --------------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
HISTORIAL_FILE = "historial.json"
MAX_HISTORIAL = 5         # máximo de mensajes guardados por usuario
TEMPERATURE = 0.7         # creatividad de la IA
MAX_RESPONSE_LENGTH = 50  # tokens máximos de respuesta
MAX_PROMPT_LENGTH = 200   # tokens máximos del prompt

bot = Bot(token=TELEGRAM_TOKEN)
app = Flask(__name__)

# --------------------------
# CARGAR MODELO PEQUEÑO (tiny-gpt2)
# --------------------------
print("Cargando modelo Tiny GPT2 (muy ligero)...")
tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
device = "cpu"  # Render Free no tiene GPU
model.to(device)
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
        historial[str(user_id)] = [{"role": "system",
                                    "content": "Eres un asistente amigable que puede mantener conversaciones de todo tipo."}]
    historial[str(user_id)].append({"role": rol, "content": mensaje})
    if len(historial[str(user_id)]) > MAX_HISTORIAL + 1:
        historial[str(user_id)] = [historial[str(user_id)][0]] + historial[str(user_id)][-MAX_HISTORIAL:]
    guardar_historial(historial)

def obtener_historial(user_id):
    historial = cargar_historial()
    return historial.get(str(user_id), [])

def resetear_historial(user_id):
    historial = cargar_historial()
    historial[str(user_id)] = [{"role": "system",
                                "content": "Eres un asistente amigable que puede mantener conversaciones de todo tipo."}]
    guardar_historial(historial)

# --------------------------
# FUNCIÓN PARA RESPONDER
# --------------------------
def gpt_responder(user_id, mensaje):
    agregar_mensaje(user_id, "user", mensaje)
    historial_usuario = obtener_historial(user_id)
    
    # Construir prompt
    prompt = ""
    for m in historial_usuario:
        if m["role"] != "system":
            prompt += m["role"] + ": " + m["content"] + "\n"
    prompt += "assistant:"

    # Tokenizar y truncar para no superar MAX_PROMPT_LENGTH
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LENGTH).to(device)
    
    # Generar respuesta sin gradientes (reduce memoria)
    with torch.no_grad():
        outputs = model.generate(**inputs,
                                 max_length=inputs.input_ids.shape[1] + MAX_RESPONSE_LENGTH,
                                 pad_token_id=tokenizer.eos_token_id,
                                 do_sample=True,
                                 temperature=TEMPERATURE)
    
    texto_respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    respuesta_final = texto_respuesta[len(prompt):].strip()
    agregar_mensaje(user_id, "assistant", respuesta_final)
    return respuesta_final

# --------------------------
# WEBHOOK DE TELEGRAM
# --------------------------
@app.route("/webhook", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    
    if update.message and update.message.text:
        user_id = update.message.from_user.id
        mensaje_usuario = update.message.text.strip()
        
        if mensaje_usuario.lower() == "/reset":
            resetear_historial(user_id)
            bot.send_message(chat_id=update.message.chat.id,
                             text="✅ Historial reiniciado. Empecemos de nuevo.")
        else:
            respuesta = gpt_responder(user_id, mensaje_usuario)
            bot.send_message(chat_id=update.message.chat.id, text=respuesta)
    
    return "ok"
@app.route("/test", methods=["GET"])
def test():
    return "ok from Flask!"

# --------------------------
# INICIAR SERVIDOR FLASK
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Servidor Flask ejecutándose en puerto {port}...")
    app.run(host="0.0.0.0", port=port)


