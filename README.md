# Multimodal Math Mentor

Multimodal Math Mentor is a math problem-solving application that supports **text, image, and audio inputs**.  
The system is designed with a focus on **structured reasoning, verification, and safety**, rather than simply generating answers.

It uses a **multi-agent pipeline** with retrieval-augmented generation (RAG), confidence-based checks, and optional human-in-the-loop (HITL) escalation to handle ambiguous or low-confidence cases.

---

## Features

- **Multiple input modes**
  - Text-based math problems
  - Image input using OCR
  - Audio input using speech-to-text

- **Structured problem solving**
  - Problem parsing and classification
  - Strategy routing based on problem complexity
  - Step-by-step solution generation

- **Retrieval-Augmented Generation (RAG)**
  - Formula and concept retrieval from a curated knowledge base
  - Vector similarity search with configurable thresholds
  - Explicit handling of retrieval failures to avoid hallucination

- **Verification & Safety**
  - Independent verification of solutions
  - Confidence scoring at each stage
  - Human-in-the-loop escalation for ambiguous or low-confidence results

- **Transparency**
  - Execution trace showing each stage of the pipeline
  - Persistent session memory for analysis and pattern reuse

---

## High-Level Flow

1. User submits a problem (text, image, or audio)
2. Input is normalized and confidence-scored
3. The problem is parsed into a structured format
4. A routing agent selects an appropriate solving strategy
5. Relevant knowledge is retrieved when required
6. A solver agent generates a step-by-step solution
7. A verifier agent checks correctness and confidence
8. An explainer agent produces a student-friendly explanation
9. Results and execution trace are returned to the UI

---

## Tech Stack

- **Frontend:** Streamlit  
- **LLMs:** Groq (Mixtral and compatible models)  
- **Frameworks:** LangChain, Pydantic  
- **Vector Store:** FAISS  
- **Embeddings:** Sentence Transformers  
- **Persistence:** SQLite  
- **Deployment:** Streamlit Cloud  

---

## Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/multimodal-math-mentor.git
cd multimodal-math-mentor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file (do not commit this file):

```env
GROQ_API_KEY=your_api_key_here
```

### 4. Run the application
```bash
streamlit run ui/app.py
```

---

## Configuration

All configuration is centralized in `src/config.py`, including:

- Model selection and temperature per agent
- RAG parameters (top-k, similarity threshold)
- OCR / ASR confidence thresholds
- HITL enable/disable flags
- Paths for databases and vector stores

This allows behavior to be tuned without modifying multiple files.

---

## Notes & Trade-offs

- The knowledge base is intentionally small and curated to reduce hallucination risk.
- Memory-based pattern reuse is kept simple (topic + difficulty) to avoid over-engineering.
- SQLite is used for persistence to keep the system lightweight and easy to deploy.
- OCR and ASR components are modular and can be swapped based on deployment constraints.

---

## Demo & Deployment

- **Live App:** Deployed using Streamlit Cloud  
- **Demo Video:** End-to-end walkthrough and brief code overview  

(Links provided separately during submission.)

---

## License

This project is intended for evaluation and demonstration purposes.
