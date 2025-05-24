import streamlit as st
import spacy
from spacy.pipeline import EntityRuler
import json
import fitz  # PyMuPDF for robust PDF text extraction
from transformers import AutoTokenizer, BertForTokenClassification
import torch

# Load BERT model (optional, if selected)
@st.cache_resource
def load_bert_model():
    model = BertForTokenClassification.from_pretrained("ner_resume_model_balanced")
    tokenizer = AutoTokenizer.from_pretrained("ner_resume_model_balanced")
    label_map = model.config.id2label
    return model, tokenizer, label_map

# Load spaCy rule-based engine
@st.cache_resource
def load_spacy_engine():
    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", before="ner")

    patterns = [
        {"label": "Name", "pattern": [{"IS_TITLE": True}, {"IS_TITLE": True}]},
        {"label": "Designation", "pattern": [{"LOWER": "developer"}]},
        {"label": "Company", "pattern": [{"IS_TITLE": True}]},
        {"label": "Email", "pattern": [{"TEXT": {"REGEX": r".+@.+\..+"}}]},
        {"label": "Location", "pattern": [{"ENT_TYPE": "GPE"}]},
    ]
    ruler.add_patterns(patterns)
    return nlp

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

# Predict using spaCy
def predict_spacy(text):
    nlp = load_spacy_engine()
    doc = nlp(text)
    entities = [{"label": ent.label_, "text": ent.text} for ent in doc.ents]
    return entities

# Predict using BERT
def predict_bert(text):
    model, tokenizer, label_map = load_bert_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    predicted_labels = [label_map[int(label)] for label in predictions[0]]

    entities = []
    current = {"label": None, "text": ""}
    for token, label in zip(tokens, predicted_labels):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        if label.startswith("B-"):
            if current["text"]:
                entities.append(current)
            current = {"label": label[2:], "text": token}
        elif label.startswith("I-") and current["label"] == label[2:]:
            current["text"] += " " + token
        else:
            if current["text"]:
                entities.append(current)
            current = {"label": None, "text": ""}
    if current["text"]:
        entities.append(current)

    return entities

# Streamlit app
st.set_page_config(page_title="Resume Entity Extractor", page_icon=":brain:", layout="centered")

st.title("🧠 Resume Entity Extractor")
ner_engine = st.radio("Select NER Engine:", ["spaCy (Rule-based)", "BERT (Trained)"])

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
resume_text = ""

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.text_area("📝 Preview Extracted Text", resume_text[:1500])

# Allow manual text input
manual_text = st.text_area("Or paste resume text manually:", value=resume_text)

if manual_text.strip():
    if ner_engine == "spaCy (Rule-based)":
        extracted_entities = predict_spacy(manual_text)
    else:
        extracted_entities = predict_bert(manual_text)

    st.subheader("📋 Extracted Entities:")
    if extracted_entities:
        st.json(extracted_entities)
        json_data = json.dumps(extracted_entities, indent=2)
        st.download_button("⬇️ Download JSON", json_data, file_name="resume_entities.json", mime="application/json")
    else:
        st.warning("⚠️ No entities found. Try pasting simpler or clearer resume text.")
