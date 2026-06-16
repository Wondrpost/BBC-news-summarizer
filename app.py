import time
import numpy as np
import streamlit as st



st.set_page_config(
    page_title="BBC News Summarizer",
    layout="wide",
    initial_sidebar_state="expanded",
)



@st.cache_resource(show_spinner=False)
def setup_nltk():
    import nltk

    for paket in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{paket}")
        except LookupError:
            try:
                nltk.download(paket, quiet=True)
            except Exception:
                pass


setup_nltk()


# css
BBC_RED = "#B01E23"

st.markdown(
    f"""
    <style>
    /* Background */
    .stApp {{
        background-color: #ffffff;
        color: #111111;
    }}
    .stApp, .stApp p, .stApp label, .stApp li, .stApp span {{
        color: #111111;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: #ececed;
        border-right: 1px solid #d4d4d4;
    }}
    section[data-testid="stSidebar"] * {{
        color: #111111;
    }}
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {{
        font-weight: 600;
        color: #111111;
    }}


    /* Input Box */
    .stTextArea textarea {{
        background-color: #f4f4f6 !important;
        color: #111111 !important;
        border: 1px solid #c8c8cc !important;
    }}
    .stTextArea textarea::placeholder {{
        color: #666666 !important;
    }}
    div[data-baseweb="select"] > div {{
        background-color: #f4f4f6 !important;
        color: #111111 !important;
        border: 1px solid #c8c8cc !important;
    }}
    div[data-baseweb="select"] * {{
        color: #111111 !important;
    }}

    /* Output Box */
    .summary-box {{
        background-color: #e2e2e4;
        border: 1px solid #c8c8cc;
        border-left: 5px solid {BBC_RED};
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        font-size: 1.05rem;
        line-height: 1.7;
        color: #111111;
    }}

    /* Metric */
    div[data-testid="stMetric"] {{
        background-color: #e2e2e4;
        border: 1px solid #c8c8cc;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }}
    div[data-testid="stMetricLabel"] {{
        color: #333333;
    }}
    div[data-testid="stMetricValue"] {{
        color: {BBC_RED};
        font-weight: 700;
    }}

    /* Button */
    .stButton > button {{
        background-color: {BBC_RED};
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.4rem;
        font-weight: 600;
    }}
    .stButton > button:hover {{
        background-color: #8c161a;
        color: #ffffff;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    f"""
    <div style="border-bottom:4px solid {BBC_RED}; padding:0.4rem 0 0.9rem 0; margin-bottom:1.8rem;">
        <h1 style="font-size:4.5rem !important; font-weight:800 !important; color:#111111 !important;
                   line-height:1.05 !important; letter-spacing:-1.5px; margin:0 !important;
                   padding:0 !important; font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
            <span style="color:{BBC_RED} !important; font-weight:900 !important;">BBC</span> News Summarizer
        </h1>
        
    </div>
    """,
    unsafe_allow_html=True,
)



def custom_textrank_with_bias(text, num_sentences=3, damping_factor=0.85, position_bonus=1e-6):
    from nltk.tokenize import sent_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx

    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    np.fill_diagonal(similarity_matrix, 0)

    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph, alpha=damping_factor)

    for i in range(min(2, len(sentences))):
        scores[i] += position_bonus * (1.5 if i == 0 else 1.0)

    ranked_sentences = sorted(
        ((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True
    )
    top_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: x[2])
    return " ".join([sentence for _, sentence, _ in top_sentences])


def custom_mmr_summarizer(text, num_sentences=3, lambda_param=0.5, position_bonus=0.0001):
    """Maximal Marginal Relevance (MMR) dengan bias posisi pada dua kalimat pertama."""
    from nltk.tokenize import sent_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    doc_vector = np.asarray(tfidf_matrix.mean(axis=0))
    relevance_scores = cosine_similarity(tfidf_matrix, doc_vector).flatten()

    for i in range(min(2, len(sentences))):
        relevance_scores[i] += position_bonus * (1.5 if i == 0 else 1.0)

    unselected = list(range(len(sentences)))
    selected = []

    first_idx = int(np.argmax(relevance_scores))
    selected.append(first_idx)
    unselected.remove(first_idx)

    while len(selected) < num_sentences and len(unselected) > 0:
        mmr_scores = {}
        selected_matrix = tfidf_matrix[np.array(selected)]

        for i in unselected:
            rel = relevance_scores[i]
            sim_to_selected = cosine_similarity(tfidf_matrix[i], selected_matrix).flatten()
            redundancy = np.max(sim_to_selected)
            mmr_scores[i] = lambda_param * rel - (1 - lambda_param) * redundancy

        best_idx = max(mmr_scores, key=mmr_scores.get)
        selected.append(best_idx)
        unselected.remove(best_idx)

    selected.sort()
    return " ".join([sentences[i] for i in selected])


@st.cache_resource(show_spinner=False)
def load_abstractive_model(model_id):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def abstractive_summarize(text, model_id, is_t5=False):
    import torch

    tokenizer, model, device = load_abstractive_model(model_id)

    input_text = ("summarize: " + text) if is_t5 else text

    inputs = tokenizer(
        input_text, max_length=1024, truncation=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


MODELS = {
    "Finetuned DistilBART (Abstraktif)": {
        "tipe": "abstraktif",
        "model_id": "nikolasone/distilbart-cnn-bbc-summarizer-finetuned",
        "is_t5": False,
    },
    "Finetuned T5-Base (Abstraktif)": {
        "tipe": "abstraktif",
        "model_id": "nikolasone/t5-base-bbc-summarizer-finetuned",
        "is_t5": True,
    },
    "TextRank with Bias (Ekstraktif)": {
        "tipe": "ekstraktif",
        "fungsi": custom_textrank_with_bias,
    },
    "MMR with Bias (Ekstraktif)": {
        "tipe": "ekstraktif",
        "fungsi": custom_mmr_summarizer,
    },
}


with st.sidebar:
    st.markdown("### Pengaturan")

    pilihan_model = st.selectbox(
        "Pilih Model Peringkas",
        options=list(MODELS.keys()),
        index=0,
        help="Pilih salah satu dari empat model peringkas berita yang tersedia.",
    )

    konfig = MODELS[pilihan_model]
    is_ekstraktif = konfig["tipe"] == "ekstraktif"
    is_mmr = pilihan_model.startswith("MMR")

    num_sentences = 3

    if is_ekstraktif:
        num_sentences = st.slider(
            "Jumlah Kalimat Ringkasan",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
            help="Banyaknya kalimat yang diekstrak untuk membentuk ringkasan.",
        )

    st.markdown("---")
    with st.expander("Tentang model"):
        st.markdown(
            """
            - **DistilBART / T5-Base**: model *deep learning* yang menghasilkan
              ringkasan **abstraktif** (menyusun kalimat baru).
            - **TextRank / MMR with Bias**: metode *machine learning* yang
              menghasilkan ringkasan **ekstraktif** (memilih kalimat asli),
              dengan *bias* posisi pada kalimat pembuka berita.
            """
        )


st.markdown("#### Teks Berita")

teks_input = st.text_area(
    label="Masukkan teks berita yang ingin diringkas",
    height=280,
    placeholder="Tempelkan artikel berita berbahasa Inggris di sini...",
    label_visibility="collapsed",
)

tombol = st.button("Ringkas Berita", type="primary")

st.markdown("---")

if tombol:
    if not teks_input or not teks_input.strip():
        st.warning("Mohon masukkan teks berita terlebih dahulu sebelum meringkas.")
    else:
        try:
            with st.spinner(f"Memproses dengan **{pilihan_model}**..."):
                mulai = time.perf_counter()

                if konfig["tipe"] == "abstraktif":
                    ringkasan = abstractive_summarize(
                        teks_input.strip(),
                        model_id=konfig["model_id"],
                        is_t5=konfig["is_t5"],
                    )
                else:
                    fungsi = konfig["fungsi"]
                    if is_mmr:
                        ringkasan = fungsi(
                            teks_input.strip(),
                            num_sentences=num_sentences,
                        )
                    else:
                        ringkasan = fungsi(
                            teks_input.strip(),
                            num_sentences=num_sentences,
                        )

                waktu_proses = time.perf_counter() - mulai

            st.markdown("#### Hasil Ringkasan")
            st.markdown(
                f'<div class="summary-box">{ringkasan}</div>',
                unsafe_allow_html=True,
            )

            jumlah_kata_asli = len(teks_input.split())
            jumlah_kata_ringkasan = len(ringkasan.split())
            rasio_kompresi = (
                (1 - jumlah_kata_ringkasan / jumlah_kata_asli) * 100
                if jumlah_kata_asli > 0
                else 0.0
            )

            st.markdown("#### Metrik")
            col1, col2, col3 = st.columns(3)
            col1.metric("Waktu Pemrosesan", f"{waktu_proses:.2f} dtk")
            col2.metric("Panjang Ringkasan", f"{jumlah_kata_ringkasan} kata")
            col3.metric("Rasio Kompresi", f"{rasio_kompresi:.1f}%")

        except ImportError:
            st.error(
                "Pustaka **transformers** dan **torch** belum terpasang. "
                "Jalankan `pip install -r requirements.txt` untuk menggunakan "
                "model abstraktif (DistilBART / T5)."
            )
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses ringkasan: {e}")
