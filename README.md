# 💬 AI-Based Coaching System (GROW Model-Driven)

> ⚠️ **Work in Progress:** This project is currently under active development. Features and structure are subject to change.

## 🧠 Project Overview

This project aims to build an **AI-driven coaching system** that facilitates structured, goal-oriented conversations using the **GROW model** (Goal, Reality, Options, Will). The system is designed to assist in coaching scenarios by understanding the coachee's emotional state, learning style, interest level, and topic adherence—ultimately generating actionable insights at the end of each session.

## 🎯 Key Objectives

- Facilitate AI-based coaching conversations using the GROW framework.
- Detect **real-time emotions**, including **sarcasm**, and **interest level** using multimodal inputs (text, audio, facial expressions).
- Identify and handle **digressions** in conversation.
- Align coaching flow with the coachee’s **preferred learning style (PLS)** (Visual, Auditory, Kinesthetic).
- Provide **post-session assessments** summarizing the coachee’s engagement, progress, and key action points.

## 🧹 Features (Planned/Implemented)

| Feature | Status |
|--------|--------|
| Transcription + Speaker Diarization | 🔄 In Progress |
| Emotion Detection (Text) | 🔄 In Progress |
| Sarcasm and Stress Detection | ✅ Implemented |
| Topic Tracking & Digression Handling | ✅ Implemented|
| PLS Classification | ✅ Implemented |
| GROW Model Annotation & Phase Tracking | ✅ Implemented |
| Session Summary Generation | 🔄 Planned |

## 🛠️ Tech Stack (to change with progression)

- **Language:** Python
- **Frontend:** Streamlit
- **Libraries/Tools:**  
  - `Whisper`, `pyannote-audio` for speech recognition and diarization  
  - `OpenCV`, `FER` for facial emotion recognition  
  - `Transformers`, `TimeSformer`, `CNN`, `RNN`, and `LSTM` models  
  - `ChromaDB`, `Gemini API` for GenAI-based session enrichment  
  - `Socket.IO` or WebSockets for real-time streaming  (tbd)

## 🚀 Getting Started

> Instructions for setting up the environment and running the app will be added as development progresses.


## 📌 Roadmap

- [x] Collect and prepare coaching session data.
- [x] Implement diarization and transcription pipeline.
- [x] Build multimodal emotion detection pipeline.
- [x] Develop GROW model annotation module.
- [x] Build live coaching session handler.
- [x] Generate coaching session summaries using GenAI.

## 🧪 Current Limitations

- Limited generalizability due to early dataset constraints.
- Interest level and digression detection are experimental

## 🤝 Contributing

Contributions are welcome! Reach out if you are interested in collaborating on this novel coaching AI system.

## 📜 License

MIT License *(to be confirmed)*

