from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from test_chatbot import chat_with_fallbacks 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str

@app.post("/chat")
async def chat(message: Message):
    try:
        response = chat_with_fallbacks(message.message)
        return {"response": response}
    except Exception as e:
        return {"response": f"Error generating response: {e}"}
