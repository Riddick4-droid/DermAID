# DermAID
# DermAId – Medical AI Agent for Dermatology: Design Blueprint

**Date:** 2026-07-25  
**Author:** Riddick Mensah
**Role:** Senior AI Engineer 
**background:** MSc. Artificial Intelligence and Data Science | Certified AI Engineer

---

## 1. Overview

DermAId is a medical AI agent that reasons through dermatological user queries using text, audio, image, or video inputs. It combines a fine‑tuned vision model, a document‑parsed knowledge base, and a retrieval‑augmented generation (RAG) pipeline orchestrated by an LLM. The agent exhibits session memory, intelligent topic switching, and a tunable strictness guardrail to prevent hallucination.

## 2. Core Requirements

- Accept multimodal input: text (typed or transcribed from audio), images, or video keyframes.
- Classify skin conditions from images using a fine‑tuned ViT‑B model; output confidence score.
- Parse a single‑PDF medical knowledge base using Landing AI’s Document Transformer.
- Store parsed chunks in a local vector database (Chroma) for semantic retrieval.
- Orchestrate retrieval and response synthesis via an LLM (OpenAI GPT‑4o) using LangChain.
- Support comparison queries between two conditions using LlamaIndex’s query engine.
- Maintain per‑session conversation history in a persistent JSON‑based KV cache.
- Configurable sliding window for history length and a strictness parameter (0–1) for grounding.
- Confidence threshold for vision predictions; uncertain cases are communicated but do not block retrieval.

## 3. Architectural Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Vision Model | ViT‑Base (fine‑tuned on DermNet 10‑class subset) | Classify skin images, output top‑1 class + confidence |
| Document Parsing | Landing AI Document AI API | One‑time PDF parsing → structured text chunks |
| Vector Database | Chroma (persistent local) | Store embedded chunks for semantic search |
| Embeddings | `text-embedding-3-small` (OpenAI) or `all-MiniLM-L6-v2` (free local) | Vectorize document chunks and queries |
| LLM Orchestrator | OpenAI GPT‑4o via LangChain Agent | Main reasoning, tool calling, answer synthesis |
| Retrieval (simple) | LangChain Chroma retriever | Standard top‑k retrieval for single‑condition queries |
| Retrieval (comparison) | LlamaIndex ComparisonQueryEngine | Two‑condition retrieval, differential synthesis |
| Memory | Custom LangChain `BaseChatMessageHistory` backed by JSON files | Persistent session history (KV cache) |
| Audio Input | Browser Web Speech API (client‑side) | Transcribe voice to text; no server‑side ASR |
| Video Handling | OpenCV keyframe extraction (server‑side) | Extract sharpest frame for vision model |
| Frontend | React / Streamlit / Gradio | Multimodal input UI; sends POST `/query` |

## 4. Design Decisions & Rationale

### 4.1 Vision Model
- **Choice:** Fine‑tune `google/vit-base-patch16-224` on DermNet’s 10‑class subset (acne, actinic keratosis, atopic dermatitis, basal cell carcinoma, melanoma, psoriasis, rosacea, seborrheic keratosis, urticaria, warts).
- **Why ViT‑B?** State‑of‑the‑art transformer for vision with strong transfer learning; aligns with Senior AI Engineer expectations (production‑style fine‑tuning, not training from scratch).
- **Confidence Handling:** Softmax probability < 0.85 → agent says “I’m not fully confident about the image, but here’s what I found based on your description.” Still retrieves from KB using text query alone.

### 4.2 Knowledge Base & Parsing
- **Content:** Synthetic concise clinical summary PDF covering 10 conditions (self‑generated). Structure: clinical features, diagnostic clues, management.
- **Parsing:** Landing AI Document AI API (free/small tier) extracts text blocks with headings. One‑time job; output stored as JSON.
- **Why Landing AI?** Meets portfolio requirement for “Document Pretrained Transformer” integration; handles layout/OCR professionally without building from scratch.

### 4.3 Retrieval Pipeline
- **Split:** Landing AI (parse) → Chroma (embed + search). Landing AI does not provide semantic search, so we decouple responsibilities.
- **Vector Store:** Chroma, persisted to local disk (`chroma_db/`). Supports LangChain and LlamaIndex native integration.
- **Chunking:** Each condition section is a separate chunk with metadata `{condition, heading, source}`.

### 4.4 Agent Orchestration – Single Orchestrator with Tools
- **Architecture:** One LangChain `AgentExecutor` (GPT‑4o) with a toolbelt:
  - `analyze_image(image_path)` – runs ViT model, returns predicted class + confidence.
  - `simple_retrieve(query)` – returns top‑k chunks from Chroma.
  - `compare_conditions(condition_a, condition_b)` – uses LlamaIndex ComparisonQueryEngine.
- **Why not parallel agents?** Steps are sequential and dependent; a single orchestrator keeps state management simple and debuggable, while still demonstrating advanced tool use.
- **Intent Routing:** The agent’s system prompt includes few‑shot examples to detect `new_topic`, `follow_up`, or `comparison` intent from the user query + history. It then chooses the appropriate tool(s). No separate classifier needed.

### 4.5 LangChain + LlamaIndex Coexistence
- **LangChain role:** Agent framework, memory, prompt templates, tool definitions.
- **LlamaIndex role:** Only used inside the `compare_conditions` tool for its built‑in `ComparisonQueryEngine`, which decomposes the query, retrieves for two conditions, and synthesises a differential.
- **No overlap:** LangChain handles conversation, memory, and tool calling; LlamaIndex is a specialised retrieval backend, not a competing orchestrator.

### 4.6 Memory & Session Management
- **Per‑session JSON KV cache:** `sessions/{session_id}.json`
  - Structure:
    ```json
    {
      "session_id": "<uuid>",
      "created_at": "<iso-timestamp>",
      "history": [
        {"role": "user", "content": "...", "timestamp": "..."},
        {"role": "assistant", "content": "...", "timestamp": "..."}
      ]
    }