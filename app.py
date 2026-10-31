"""LangScout - Streamlit web app.   Run with:  streamlit run app.py"""
import io
import json

import pandas as pd
import streamlit as st

from langscout.config import REPORT_DIR
from langscout.detector import OK, LanguageDetector
from langscout.explain import highlight_html

st.set_page_config(page_title="LangScout - Language Detection", page_icon="🌍", layout="centered")

EXAMPLES = {
    "English": "The weather is lovely today, let's go for a walk.",
    "Français": "Ceci est une belle journée pour se promener.",
    "हिन्दी": "यह एक अच्छा दिन है, चलो बाहर घूमने चलते हैं।",
    "தமிழ்": "இன்று வானிலை மிகவும் நன்றாக உள்ளது.",
    "Deutsch": "Ich lerne gerade Python und maschinelles Lernen.",
    "Hinglish": "kal main market jaunga, tum bhi chalo",
    "Mixed": "I love this city. Me encanta esta ciudad. Ich liebe diese Stadt.",
}


@st.cache_resource(show_spinner="Loading model ...")
def load_detector() -> LanguageDetector:
    return LanguageDetector()


@st.cache_data
def load_report(name: str):
    path = REPORT_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")) if name.endswith(".json") else pd.read_csv(path)


detector = load_detector()

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence threshold", 0.0, 0.99, float(detector.threshold), 0.01,
                          help="Predictions below this confidence are flagged as uncertain instead of being trusted blindly.")
    detector.threshold = threshold
    st.caption(f"Model: **{detector.metadata.get('model', 'n/a')}**")
    with st.expander(f"Supported languages ({len(detector.languages)})"):
        st.write(", ".join(detector.languages))

st.title("🌍 LangScout")
st.write("Detects **17 languages plus Hinglish** (Hindi typed in English letters). Unsupported scripts, gibberish "
         "and low-confidence inputs are flagged instead of getting a confident wrong answer.")

tab_detect, tab_mixed, tab_batch, tab_model = st.tabs(["Detect", "Mixed text", "Batch CSV", "About the model"])


def show_explanation(text: str) -> None:
    info = detector.explain(text)
    if info is None:
        return
    with st.expander(f"Why {info['target']} and not {info['versus']}?"):
        st.markdown(highlight_html(info["words"]), unsafe_allow_html=True)
        st.caption(f"Green words push towards {info['target']}, red words towards {info['versus']}. "
                   f"Total evidence margin: {info['margin']:.2f} (log-likelihood units).")
        left, right = st.columns(2)
        left.write(f"**Character patterns for {info['target']}**")
        left.table(pd.DataFrame(info["favours_target"], columns=["n-gram", "score"]).set_index("n-gram"))
        right.write(f"**Character patterns for {info['versus']}**")
        if info["favours_versus"]:
            right.table(pd.DataFrame(info["favours_versus"], columns=["n-gram", "score"]).set_index("n-gram"))
        else:
            right.caption("Nothing in this text favours the runner-up.")


def show_prediction(p) -> None:
    if p.language is None:
        st.error(f"**{p.status.replace('_', ' ').title()}** - {p.note}")
        return
    if p.is_confident:
        st.success(f"Detected language: **{p.language}**  ·  confidence **{p.confidence:.1%}**")
    else:
        st.warning(f"Uncertain - best guess: **{p.language}**  ·  confidence **{p.confidence:.1%}**")
    if p.note:
        st.caption(p.note)
    st.caption(f"Script: {p.script}  ·  decided by: {'writing system' if p.method == 'script' else 'statistical model'}")
    if len(p.top_k) > 1:
        st.bar_chart(pd.DataFrame(p.top_k, columns=["Language", "Probability"]).set_index("Language"))


with tab_detect:
    choice = st.selectbox("Try an example", ["(type your own)"] + list(EXAMPLES))
    default = "" if choice == "(type your own)" else EXAMPLES[choice]
    text = st.text_area("Text", value=default, height=120, placeholder="यह एक अच्छा दिन है", key=f"text_{choice}")
    if st.button("Detect language", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            show_prediction(detector.detect(text))
            show_explanation(text)

with tab_mixed:
    st.write("Each sentence or line is analysed separately - useful for multilingual paragraphs.")
    mixed = st.text_area("Mixed-language text", value=EXAMPLES["Mixed"], height=140, key="mixed")
    if st.button("Analyse sentences"):
        segs = detector.detect_segments(mixed)
        if not segs:
            st.warning("Please enter some text.")
        for s in segs:
            label = s.language or s.status.replace("_", " ")
            st.markdown(f"**{label}** ({s.confidence:.0%})" + ("" if s.is_confident else " ⚠️") + f" - {s.text}")

with tab_batch:
    st.write("Upload a CSV, choose the text column, and download the predictions.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        frame = pd.read_csv(up)
        col = st.selectbox("Column containing the text", frame.columns)
        if st.button("Run batch detection"):
            preds = detector.detect_batch(frame[col].fillna("").astype(str).tolist(), top_k=1)
            out = frame.assign(
                detected_language=[p.language for p in preds],
                confidence=[round(p.confidence, 4) for p in preds],
                status=[p.status for p in preds],
            )
            st.dataframe(out, use_container_width=True)
            st.caption(f"{sum(p.status == OK for p in preds)}/{len(preds)} predictions above the confidence threshold.")
            buf = io.StringIO(); out.to_csv(buf, index=False)
            st.download_button("Download results", buf.getvalue(), "langscout_predictions.csv", "text/csv")

with tab_model:
    meta = detector.metadata
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{meta.get('test_accuracy', 0):.1%}")
    c2.metric("Test macro-F1", f"{meta.get('test_macro_f1', 0):.1%}")
    c3.metric("Out-of-domain samples", f"{meta.get('everyday_accuracy', 0):.0%}")
    comparison = load_report("model_comparison.csv")
    if comparison is not None:
        st.subheader("Model comparison")
        st.dataframe(comparison, use_container_width=True)
    for img, cap in [("short_text_robustness.png", "Accuracy vs. input length"),
                     ("confusion_matrix.png", "Confusion matrix (test set)"),
                     ("per_class_f1.png", "F1 per language")]:
        if (REPORT_DIR / img).exists():
            st.image(str(REPORT_DIR / img), caption=cap)
    st.subheader("Limitations")
    st.markdown(
        "- Only the 17 listed languages; others are mapped to the closest one (or flagged if the script is unknown).\n"
        "- Hinglish is detected, but spellings vary a lot between people and Roman Urdu looks almost the same.\n"
        "- Training data is Wikipedia-style text; very short, slangy or code-switched text is less reliable.\n"
        "- Arabic / Cyrillic / Devanagari are shared with languages outside the 17 (Persian, Ukrainian, Marathi ...)."
    )
