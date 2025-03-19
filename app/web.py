"""
Web interface for interacting with Thallan NPC.
"""
import os
import sys
from typing import Dict
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.character import CharacterPersona


# Initialize FastAPI app
app = FastAPI(title="Thallan - Radiant Citadel NPC")

# Set up templates and static files
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

# Create directories if they don't exist
templates_dir.mkdir(exist_ok=True)
static_dir.mkdir(exist_ok=True)

# Set up templates
templates = Jinja2Templates(directory=str(templates_dir))

# Create the character instance
character = CharacterPersona()

# Global chat history
chat_histories = {}


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    # Create HTML template if it doesn't exist
    index_path = templates_dir / "index.html"
    if not index_path.exists():
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Thallan - Radiant Citadel NPC</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .chat-container {
            height: calc(100vh - 160px);
            overflow-y: auto;
        }
        .user-message {
            background-color: #e2f1ff;
            border-radius: 20px 20px 4px 20px;
        }
        .npc-message {
            background-color: #fffde2;
            border-radius: 20px 20px 20px 4px;
        }
    </style>
</head>
<body class="bg-gray-100 text-gray-900">
    <div class="container mx-auto p-4">
        <div class="bg-white rounded-lg shadow-lg p-6">
            <div class="mb-4 flex items-center">
                <img src="https://via.placeholder.com/50" alt="Thallan" class="rounded-full mr-3">
                <div>
                    <h1 class="text-2xl font-bold">Thallan</h1>
                    <p class="text-sm text-gray-600">Scholar and Guide of the Radiant Citadel</p>
                </div>
            </div>

            <div class="chat-container border border-gray-300 rounded-lg p-4 mb-4" id="chat-messages">
                <div class="npc-message p-3 mb-4 shadow-sm">
                    <p>Greetings, traveler! I am Thallan, a scholar and guide of the Radiant Citadel. How may I assist you on your journey today?</p>
                </div>
            </div>

            <div class="flex">
                <input type="text" id="message-input" placeholder="Type your message..." 
                       class="flex-grow p-3 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-yellow-400">
                <button id="send-button" class="bg-yellow-500 hover:bg-yellow-600 text-white px-6 py-3 rounded-r-lg">
                    Send
                </button>
            </div>
        </div>
    </div>

    <script>
        let ws;
        const chatMessages = document.getElementById('chat-messages');
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');

        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onopen = function() {
                console.log('WebSocket connection established');
            };

            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                addMessage(message.content, message.sender);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            };

            ws.onclose = function() {
                console.log('WebSocket connection closed');
                // Try to reconnect after a delay
                setTimeout(connect, 1000);
            };
        }

        function addMessage(content, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.className = sender === 'user' ? 
                'user-message p-3 mb-4 shadow-sm ml-12' : 
                'npc-message p-3 mb-4 shadow-sm mr-12';
            messageDiv.innerHTML = `<p>${content}</p>`;
            chatMessages.appendChild(messageDiv);
        }

        function sendMessage() {
            const message = messageInput.value.trim();
            if (message && ws && ws.readyState === WebSocket.OPEN) {
                addMessage(message, 'user');
                ws.send(message);
                messageInput.value = '';
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }

        // Event listeners
        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Connect when the page loads
        window.addEventListener('load', connect);
    </script>
</body>
</html>""")

    # Create CSS file if it doesn't exist
    css_path = static_dir / "styles.css"
    if not css_path.exists():
        with open(css_path, 'w', encoding='utf-8') as f:
            f.write("""/* Add custom styles here */""")


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

        # Initialize chat history for this client
        if client_id not in chat_histories:
            chat_histories[client_id] = []

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_message(self, message: str, client_id: str, sender: str = "npc"):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json({
                "content": message,
                "sender": sender
            })


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = f"client_{id(websocket)}"
    await manager.connect(websocket, client_id)

    try:
        while True:
            # Receive message from client
            user_message = await websocket.receive_text()

            # Get the chat history for this client
            history = chat_histories.get(client_id, [])

            # Generate response
            response = character.generate_response(user_message, history)

            # Update chat history
            history.append({
                "user": user_message,
                "character": response
            })
            chat_histories[client_id] = history

            # Send response back to client
            await manager.send_message(response, client_id)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        # Save chat history or cleanup


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the web server."""
    uvicorn.run("app.web:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    run_server()