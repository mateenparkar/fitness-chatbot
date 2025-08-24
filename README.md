# My AI Web App

A full-stack web application featuring a **FastAPI backend** running a **GPT-2 model** and a **React frontend** built with **Vite**. The app allows users to interact with a text generation model via a simple web interface.

## Features

- FastAPI backend serving a GPT-2 model API
- React frontend with Vite for fast, modern UI
- Dockerized setup for easy local development
- Ready for deployment on platforms like Render, Railway, or Fly.io

## Tech Stack

- **Backend:** Python, FastAPI, PyTorch, Transformers (GPT-2)
- **Frontend:** React, Vite, NPM
- **Deployment:** Docker, Nginx (for frontend in production)
- **Dev Tools:** Docker Compose for local development

## Getting Started

### Prerequisites

- Docker & Docker Compose installed
- Node.js & NPM (optional if using Docker for frontend build)

### Running Locally (Docker)

1. Clone the repo:

```bash
git clone https://github.com/your-username/fitness-chatbot.git
cd fitness-chatbot
```

2. Start the backend and frontend with Docker Compose:
   
```bash
docker-compose up --build
```

3. Access the app
```bash
Frontend: http://localhost:3000
```
