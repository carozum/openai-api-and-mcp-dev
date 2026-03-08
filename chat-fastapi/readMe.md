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
2. Le contenu transcrit est modéré avant traitement
3. Le LLM génère une réponse en streaming
4. Dès qu'une phrase est complète, elle est envoyée au TTS et lue immédiatement
5. Bouton pause/reprendre pendant la lecture

### 🛡️ Modération
Tous les contenus utilisateur passent par l'API de modération OpenAI (`omni-moderation-latest`) avant traitement, sur **tous les onglets** :

| Onglet | Ce qui est modéré |
|---|---|
| Text Chat | Message texte avant envoi au LLM |
| Image | Prompt avant génération |
| Audio — STT | Texte transcrit depuis le micro |
| Audio — Traduction | Texte traduit depuis le micro |
| Audio — TTS | Texte saisi avant synthèse vocale |
| Voice Chat | Transcription de la voix avant envoi au LLM |

En cas de contenu refusé, l'API retourne une **erreur générique** (`🚫 Ce contenu ne peut pas être traité.`) sans exposer la catégorie au client. Les détails sont loggés côté serveur uniquement.

---

## Architecture

```
chatbot-app/
├── main.py           # Backend FastAPI — routes API et streaming SSE
├── utils.py          # Fonctions OpenAI (LLM, TTS, STT, image, modération)
├── config.py         # Configuration centralisée via pydantic-settings
├── static/
│   └── index.html    # Frontend complet (HTML + CSS + JS)
├── temporary_files/  # Fichiers audio temporaires (créé automatiquement)
└── .env              # Clé API OpenAI (non versionné)
```

### Flux de données

```
[Navigateur]
    │
    ├── POST /api/chat/stream       → modération → SSE tokens LLM
    ├── POST /api/image             → modération → JSON base64
    ├── POST /api/transcribe        → STT → modération → JSON texte
    ├── POST /api/translate         → STT → modération → JSON texte
    ├── POST /api/tts               → modération → fichier WAV
    ├── POST /api/moderate          → JSON résultat modération
    └── POST /api/voice-chat/stream → STT → modération → SSE (tokens + audio WAV base64)
```

### Pipeline Modération

```
Contenu utilisateur (texte ou transcription)
        │
        ▼
_check_moderation()
        │
        ├── flagged → HTTP 422  →  message générique au client
        │                          catégories loggées côté serveur
        │
        └── ok → traitement normal (LLM / TTS / image)
```

### Pipeline Voice Chat

```
Audio micro → STT → modération → LLM stream → [phrase détectée] → TTS → audio navigateur
                                                     ↑                        ↓
                                                accumulation             lecture immédiate
                                                 du buffer              pendant que le LLM
                                                                         continue de générer
```

---

## Installation

### Prérequis

- Python 3.11+
- ffmpeg (conversion audio WebM → WAV)

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
pip install fastapi uvicorn python-multipart openai pydub pydantic-settings

# Créer le fichier .env
echo "OPENAI_API_KEY=sk-..." > .env

# Lancer le serveur
uvicorn main:app --reload
```

Ouvrir `http://localhost:8000`

---

## Configuration

Tous les paramètres par défaut sont centralisés dans `config.py` via `pydantic-settings`. Ils peuvent être surchargés par des variables d'environnement dans le fichier `.env`.

| Variable `.env` | Description | Défaut |
|---|---|---|
| `OPENAI_API_KEY` | Clé API OpenAI **(obligatoire)** | — |
| `DEFAULT_TEXT_MODEL` | Modèle LLM | `gpt-4o-mini` |
| `DEFAULT_TEMPERATURE` | Température LLM | `0.7` |
| `DEFAULT_IMAGE_MODEL` | Modèle image | `gpt-image-1` |
| `DEFAULT_IMAGE_SIZE` | Taille image | `1024x1024` |
| `DEFAULT_STT_MODEL` | Modèle transcription | `gpt-4o-transcribe` |
| `DEFAULT_TTS_MODEL` | Modèle synthèse vocale | `tts-1` |
| `DEFAULT_TTS_VOICE` | Voix TTS | `alloy` |
| `DEFAULT_TTS_SPEED` | Vitesse TTS | `1.0` |
| `SYSTEM_MESSAGE` | System prompt du LLM | *(voir config.py)* |
| `TMP_DIR` | Dossier fichiers temporaires | `temporary_files` |

Les paramètres sont également ajustables en temps réel depuis le header de l'interface.

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
| `POST` | `/api/moderate` | Modération d'un texte — retourne `{ flagged: bool }` |

### Codes de réponse

| Code | Signification |
|---|---|
| `200` | Succès |
| `422` | Contenu refusé par la modération — message générique, détails côté serveur uniquement |
| `500` | Erreur serveur (OpenAI API, ffmpeg, etc.) |

---

## Dépendances principales

| Package | Usage |
|---|---|
| `fastapi` | Framework web backend |
| `uvicorn` | Serveur ASGI |
| `openai` | SDK OpenAI (LLM, TTS, STT, image, modération) |
| `pydub` | Conversion audio WebM → WAV |
| `python-multipart` | Upload de fichiers audio |
| `pydantic-settings` | Configuration centralisée via `.env` |