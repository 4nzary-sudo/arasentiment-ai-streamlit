
import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model repository on Hugging Face
MODEL_ID = "mohammedalnzary/arabic-sentiment-arabert"

st.set_page_config(
    page_title="AraSentiment AI",
    page_icon="💬",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
}

.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: 800;
    color: #6C63FF;
    direction: rtl;
}

.subtitle {
    text-align: center;
    color: #888;
    direction: rtl;
    font-size: 16px;
}

.result-box {
    padding: 22px;
    border-radius: 15px;
    text-align: center;
    direction: rtl;
    background: rgba(108, 99, 255, 0.10);
    border: 1px solid rgba(108, 99, 255, 0.35);
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">💬 AraSentiment AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">تحليل المشاعر للنصوص العربية باستخدام AraBERT</div>',
    unsafe_allow_html=True
)

st.divider()


def get_hf_token():
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return os.getenv("HF_TOKEN")


@st.cache_resource
def load_model():
    token = get_hf_token()

    if not token:
        raise RuntimeError("HF_TOKEN is not configured.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=token
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        token=token
    )

    model.to("cpu")
    model.eval()

    return tokenizer, model


def analyze_sentiment(text):
    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    negative_score = float(probabilities[0])
    positive_score = float(probabilities[1])

    if positive_score >= negative_score:
        label = "إيجابي 😊"
        confidence = positive_score
    else:
        label = "سلبي 😞"
        confidence = negative_score

    return label, confidence, negative_score, positive_score


st.markdown("### ✍️ أدخل النص المراد تحليله")

user_text = st.text_area(
    "اكتب مراجعة أو تعليقًا باللغة العربية:",
    height=150,
    placeholder="مثال: الفندق ممتاز والخدمة رائعة والموظفون متعاونون..."
)

if st.button("🔍 تحليل المشاعر", use_container_width=True):

    if not user_text.strip():
        st.warning("يرجى كتابة نص قبل التحليل.")

    else:
        try:
            with st.spinner("جاري تحليل النص باستخدام AraBERT..."):
                label, confidence, negative, positive = analyze_sentiment(
                    user_text.strip()
                )

            st.markdown("---")
            st.markdown(
                '<div class="result-box">'
                '<h3>نتيجة التحليل</h3>'
                f'<h2>{label}</h2>'
                f'<p>نسبة الثقة: {confidence * 100:.2f}%</p>'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown("### 📊 تفاصيل النتائج")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("المشاعر الإيجابية", f"{positive * 100:.2f}%")

            with col2:
                st.metric("المشاعر السلبية", f"{negative * 100:.2f}%")

        except Exception:
            st.error("تعذر تحميل النموذج أو تحليل النص.")
            st.info(
                "تحقق من إعداد HF_TOKEN ومن صلاحية الوصول "
                "إلى مستودع النموذج في Hugging Face."
            )

st.divider()

st.markdown("""
<div style="text-align:center;direction:rtl;color:gray;">
    AraSentiment AI | Arabic NLP Project<br>
    Powered by AraBERT
</div>
""", unsafe_allow_html=True)
  
