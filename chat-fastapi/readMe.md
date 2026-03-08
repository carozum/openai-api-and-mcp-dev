# 🤖 Chatbot App

Application multimodale basée sur les APIs OpenAI, avec un backend **FastAPI** et un frontend **HTML/JS** natif.

---

## Fonctionnalités

### 💬 Text Chat
Chat multi-turn avec streaming token par token. L'historique de conversation est conservé côté client et renvoyé à chaque requête pour maintenir le contexte. Un micro permet de dicter son message (transcription automatique avant envoi).

### 🖼️ Image
Génération d'images à partir d'un prompt texte via l'API `gpt-image-1`. L'image est retournée en base64 et affichée directement dans le navigateur.

### 🎙️ Audio
Trois outils indépendants :
- **Speech → Text** : transcription vocale dans la langue d'origine (GPT-4o Transcribe ou Whisper-1)
- **Speech → Translation** : transcription + traduction en anglais (Whisper-1)
- **Text → Speech** : synthèse vocale à partir d'un texte, avec choix de la voix et de la vitesse

### 🤖 Voice Chat
Conversation vocale complète et multi-turn :
1. L'utilisateur parle → transcription automatique
2. Le LLM génère une réponse en streaming
3. Dès qu'une phrase est complète, elle est envoyée au TTS et lue immédiatement
4. Bouton pause/reprendre pendant la lecture

---

## Architecture

```
chatbot-app/
├── main.py           # Backend FastAPI — routes API et streaming SSE
├── utils.py          # Fonctions OpenAI (LLM, TTS, STT, image)
├── static/
│   └── index.html    # Frontend complet (HTML + CSS + JS)
├── temporary_files/  # Fichiers audio temporaires (créé automatiquement)
└── .env              # Clé API OpenAI (non versionné)
```

### Flux de données

```
[Navigateur]
    │
    ├── POST /api/chat/stream       → SSE tokens LLM
    ├── POST /api/image             → JSON base64
    ├── POST /api/transcribe        → JSON texte
    ├── POST /api/translate         → JSON texte
    ├── POST /api/tts               → fichier WAV
    └── POST /api/voice-chat/stream → SSE (transcription + tokens + audio WAV base64)
```

### Streaming SSE (Voice Chat)

Le voice chat utilise une pipeline phrase par phrase pour minimiser la latence :

```
Audio micro → STT → LLM stream → [phrase détectée] → TTS → audio envoyé au navigateur
                                        ↑                         ↓
                                   accumulation              lecture immédiate
                                    du buffer               pendant que le LLM
                                                             continue de générer
```

---

## Installation

### Prérequis

- Python 3.11+
- ffmpeg (pour la conversion audio)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### Setup

```bash
# Cloner le projet
git clone <repo>
cd chatbot-app

# Installer les dépendances Python
pip install fastapi uvicorn python-multipart openai pydub python-dotenv colorama

# Créer le fichier .env
echo "OPENAI_API_KEY=sk-..." > .env

# Lancer le serveur
uvicorn main:app --reload
```

Ouvrir `http://localhost:8000`

---

## Configuration

Tous les paramètres sont accessibles depuis le header de l'interface :

| Paramètre | Description | Valeurs possibles |
|---|---|---|
| Text model | Modèle LLM pour le chat | `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo` |
| Temperature | Créativité des réponses | 0.0 → 1.5 |
| Image model | Modèle de génération d'images | `gpt-image-1`, `gpt-image-1-mini` |
| STT | Modèle de transcription | `gpt-4o-transcribe`, `whisper-1` |
| Voice | Voix TTS | `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` |
| TTS model | Modèle de synthèse vocale | `tts-1`, `tts-1-hd` |
| Speed | Vitesse de lecture | 0.5 → 2.0 |

---

## API Reference

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/` | Sert le frontend |
| `POST` | `/api/chat/stream` | Chat LLM en streaming SSE |
| `POST` | `/api/image` | Génération d'image |
| `POST` | `/api/transcribe` | Transcription audio → texte |
| `POST` | `/api/translate` | Traduction audio → anglais |
| `POST` | `/api/tts` | Synthèse texte → audio WAV |
| `POST` | `/api/voice-chat/stream` | Pipeline voix complète en streaming SSE |

---

## Dépendances principales

| Package | Usage |
|---|---|
| `fastapi` | Framework web backend |
| `uvicorn` | Serveur ASGI |
| `openai` | SDK OpenAI (LLM, TTS, STT, image) |
| `pydub` | Conversion audio WebM → WAV |
| `python-multipart` | Upload de fichiers audio |
| `python-dotenv` | Chargement de la clé API |