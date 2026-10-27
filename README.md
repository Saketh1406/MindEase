# MindEase — Review 2 Starter Prototype

## What this is
A small academic MVP to demonstrate:
- A local Streamlit chat UI
- Transparent rule-based routing for broad categories
- A basic TF-IDF + cosine-similarity retrieval baseline (RAG-inspired)
- Optional OpenAI API response generation
- A clearly disclosed fallback mode if no API key is configured
- A safety escalation screen with India's official Tele-MANAS contact

## Important limitations
- This is not a clinical product or a validated risk detector.
- Keyword rules can miss context and produce false positives.
- TF-IDF is a lexical retrieval baseline, not dense embedding retrieval.
- Reinforcement learning is NOT implemented.
- There is no persistent database or anonymous analytics pipeline.
- Do not use real personal/mental-health data in the demo.
- Do not claim clinical effectiveness or present fabricated metrics.

## Setup on Windows (PowerShell)
1. Install Python 3.11 or 3.12 if it is not installed.
2. Extract the ZIP to a simple folder such as `C:\MindEase_Review2_Starter`.
3. Open that folder in VS Code.
4. Open Terminal > New Terminal and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Optional LLM API mode
The app works without an API key in fallback mode. If your team already has authorized API access, set the key in the current PowerShell session (do not paste it into code or slides):
```powershell
$env:OPENAI_API_KEY="your-key-here"
$env:OPENAI_MODEL="gpt-4o-mini"
streamlit run app.py
```
API usage may incur charges. Do not create a key or spend money just for the review if not approved by your team/guide. Without the key, clearly say that the demo is in fallback mode and does not demonstrate live LLM generation.

## Demo scenarios
1. General stress: `I've had a stressful day and feel overwhelmed.`
2. Exam anxiety: `I am anxious about my exams tomorrow.`
3. Safety routing test: use a prepared synthetic test phrase agreed by your guide; do not use real user disclosures.
4. Retrieval: ask about exam preparation or support resources, then expand the retrieved-context panel.

## Suggested Review 2 algorithm description
1. Rule-based category routing (baseline, not clinically validated).
2. TF-IDF vectorization of curated resource text.
3. Cosine similarity ranking to retrieve top-k resource passages.
4. Optional LLM response generation grounded with retrieved context.
5. Fixed test suite to evaluate routing, retrieval relevance, response quality, safety behavior, and latency.

## Performance metrics
- Classification: precision, recall, F1 per class, confusion matrix. Report the test-set size.
- Retrieval: Recall@K or Precision@K using manually labelled relevant documents; do not call raw TF-IDF similarity an accuracy score.
- Response quality: human-rated empathy/helpfulness on a 1–5 rubric, with the number of raters disclosed.
- Safety: pass/fail per predefined synthetic test case, especially missed escalation cases.
- Latency: median and 95th percentile if enough runs; otherwise report the number of runs and mean/median.
- Report actual measured values only. Do not invent results.

## Official resource
Government of India, Ministry of Health and Family Welfare, National Mental Health Programme / Tele-MANAS:
https://dghs.mohfw.gov.in/national-mental-health-programme.php
The page lists Tele-MANAS helpline 14416 and 1800-89-14416.
