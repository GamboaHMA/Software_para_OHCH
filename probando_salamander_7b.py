from ollama import chat

MODEL = 'hdnh2006/salamandra-7b-instruct:latest'

respuesta = chat(
    model=MODEL,
    messages=[
        {'role': 'system', 'content': 'Responde en español y de forma breve.'},
        {'role': 'user', 'content': 'Dame 3 usos de python.'}
    ]
)

print(respuesta.message.content)