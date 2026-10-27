import os
import time
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="MindEase | Review 2 Prototype", page_icon="🌿", layout="wide")

st.title("🌿 MindEase")
st.caption("RAG-inspired, LLM-assisted first-response wellness support — academic prototype")

st.warning(
    "Prototype disclaimer: MindEase is not a therapist, diagnostic tool, or emergency service. "
    "Do not enter real names or identifying details. This demo does not store chat messages in a database."
)

# Curated starter corpus. Team should verify/update content and sources before presenting.
RESOURCES = [
    {
        "id": "R1",
        "title": "Breaking a stressful task into smaller steps",
        "text": "For everyday stress, consider choosing one small next step, setting a short time block, "
                "and taking a brief pause between tasks. This is a general wellness suggestion, not treatment.",
        "tags": "stress overwhelmed tasks planning focus"
    },
    {
        "id": "R2",
        "title": "Exam preparation and anxiety",
        "text": "For exam anxiety, write down the topics you need to revise, prioritize the most important ones, "
                "and use short study intervals with breaks. Protect time for sleep and ask a trusted person or "
                "campus support service for help if anxiety feels difficult to manage.",
        "tags": "exam anxiety study student revision tests"
    },
    {
        "id": "R3",
        "title": "Grounding and mindful pause",
        "text": "A brief grounding pause may help some people reconnect with the present moment. Notice your feet "
                "on the floor, name a few things you can see, and take comfortable, unforced breaths. Stop if it "
                "does not feel helpful.",
        "tags": "grounding overwhelmed anxious breathing present"
    },
    {
        "id": "R4",
        "title": "Seeking support from a trusted person",
        "text": "If distress is persistent or affecting daily life, consider talking with someone you trust, "
                "a campus counsellor, or a qualified mental-health professional. You do not need to manage "
                "everything alone.",
        "tags": "support counsellor professional distress"
    },
    {
        "id": "R5",
        "title": "India Tele-MANAS mental-health support",
        "text": "India's Government Tele-MANAS service provides 24x7 tele-mental-health support. Call 14416 "
                "or 1800-89-14416. Verify service details from the Ministry of Health and Family Welfare website.",
        "tags": "india helpline professional crisis urgent support tele manas"
    }
]

# Simple, transparent keyword rules for a classroom prototype only.
# These rules are not a clinically validated risk detector.
ESCALATION_TERMS = [
    "suicid", "kill myself", "end my life", "hurt myself", "self harm",
    "harm myself", "don't want to live", "do not want to live"
]
EXAM_TERMS = ["exam", "test", "semester", "marks", "study", "revision", "viva"]
STRESS_TERMS = ["stress", "overwhelmed", "pressure", "anxious", "anxiety", "worried", "panic", "tired"]

def classify_message(message: str):
    text = message.lower()
    if any(term in text for term in ESCALATION_TERMS):
        return "escalation_required", "high"
    if any(term in text for term in EXAM_TERMS):
        return "exam_anxiety", "low"
    if any(term in text for term in STRESS_TERMS):
        return "general_stress", "low"
    return "general_support", "low"

@st.cache_resource
def build_retriever():
    docs = [f"{r['title']} {r['text']} {r['tags']}" for r in RESOURCES]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix

def retrieve_resources(query, k=2):
    vectorizer, matrix = build_retriever()
    q = vectorizer.transform([query])
    scores = cosine_similarity(q, matrix).flatten()
    ranked = scores.argsort()[::-1][:k]
    return [(RESOURCES[i], float(scores[i])) for i in ranked if scores[i] > 0]

def fallback_response(category, message, retrieved):
    if category == "exam_anxiety":
        reply = ("It sounds like exams are putting pressure on you. You could choose one topic to work on "
                 "for a short block, then take a break. What feels most manageable to start with?")
    elif category == "general_stress":
        reply = ("That sounds like a lot to carry right now. It may help to pause and choose one small next step "
                 "rather than solving everything at once. Would you like to talk about what's making today difficult?")
    else:
        reply = ("Thank you for sharing that. I can offer general wellness information, but I can't diagnose or "
                 "replace a professional. Would a practical coping idea or information about support options be useful?")
    if retrieved:
        reply += "\n\nRelevant resource: " + retrieved[0][0]["title"] + "."
    return reply

def llm_response(message, category, retrieved):
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return fallback_response(category, message, retrieved), "Fallback mode (no API key)"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        context = "\n\n".join([f"{r['title']}: {r['text']}" for r, score in retrieved])
        system_prompt = (
            "You are MindEase, a non-clinical first-response wellness assistant. Be calm, respectful, "
            "brief and nonjudgmental. Do not diagnose, prescribe medication, claim to be a therapist, or "
            "promise confidentiality. Offer one practical low-risk suggestion and invite the user to choose. "
            "Use the supplied retrieved context when relevant and do not invent helpline details. "
            "If a message may indicate immediate danger, prioritize urging the person to contact local emergency "
            "services or a trusted person now and show the verified support resource provided by the application. "
            "Do not treat yourself as a substitute for professional care."
        )
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Category: {category}\nRetrieved context:\n{context}\n\nUser message: {message}"}
            ],
            temperature=0.4,
            max_tokens=220
        )
        return response.choices[0].message.content.strip(), "LLM API mode"
    except Exception as exc:
        return fallback_response(category, message, retrieved), f"Fallback mode (API unavailable: {type(exc).__name__})"

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Prototype controls")
    st.write("**Mode:** local demo")
    st.write("**Retrieval:** TF-IDF + cosine similarity")
    st.write("**Safety routing:** transparent keyword rules")
    st.write("**Generation:** optional OpenAI API")
    st.divider()
    st.caption("No persistent database is configured. Chat history remains in this Streamlit session only.")
    if st.button("Clear demo conversation"):
        st.session_state.history = []
        st.rerun()

for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("meta"):
            st.caption(entry["meta"])

message = st.chat_input("Type a general wellness question (avoid identifying details)...")
if message:
    started = time.perf_counter()
    st.session_state.history.append({"role": "user", "content": message})
    category, level = classify_message(message)

    with st.chat_message("user"):
        st.markdown(message)

    with st.chat_message("assistant"):
        if category == "escalation_required":
            reply = (
                "This message may need immediate human support. MindEase cannot assess your safety reliably. "
                "If you may be in immediate danger, contact local emergency services now and reach out to a "
                "trusted person who can stay with you. In India, Tele-MANAS offers 24x7 mental-health support: "
                "call 14416 or 1800-89-14416."
            )
            st.error(reply)
            mode = "Rule-based escalation"
            retrieved = retrieve_resources("India mental health support helpline", k=2)
        else:
            retrieved = retrieve_resources(message, k=2)
            reply, mode = llm_response(message, category, retrieved)
            st.markdown(reply)
            if retrieved:
                with st.expander("Retrieved context (RAG baseline)"):
                    for resource, score in retrieved:
                        st.markdown(f"**{resource['title']}** — similarity: `{score:.3f}`")
                        st.write(resource["text"])
        elapsed_ms = (time.perf_counter() - started) * 1000
        meta = f"Category: {category} | Route: {level} | Mode: {mode} | Latency: {elapsed_ms:.0f} ms"
        st.caption(meta)
    st.session_state.history.append({"role": "assistant", "content": reply if category == "escalation_required" else reply, "meta": meta})

with st.expander("About this prototype"):
    st.write(
        "This is an academic MVP, not a clinically validated system. The keyword classifier can miss context "
        "or produce false alarms. TF-IDF is a lexical retrieval baseline, not embedding-based semantic search. "
        "Reinforcement learning is not implemented in this version. Do not use this prototype for real-world "
        "crisis triage or collect real users' mental-health conversations."
    )
    st.markdown("Verified resource: [Government of India — National Mental Health Programme / Tele-MANAS](https://dghs.mohfw.gov.in/national-mental-health-programme.php)")
