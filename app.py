"""Aplikasi Klasifikasi Berita: Crawling -> Preprocessing -> Skip-gram -> Naive Bayes."""
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA

import crawler
import modeling as md
import preprocessing as pp

st.set_page_config(page_title="Klasifikasi Berita", page_icon="📰", layout="wide")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RAW_CSV, CLEAN_CSV = DATA_DIR / "berita_raw.csv", DATA_DIR / "berita_clean.csv"

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero {background: linear-gradient(135deg,#1e3a8a,#6d28d9); color:white; padding:1.6rem 2rem;
       border-radius:16px; margin-bottom:1rem;}
.hero h1 {margin:0; font-size:2rem; color:white;} .hero p {margin:.3rem 0 0; opacity:.9;}
.pred {background:#f0fdf4; border:1px solid #86efac; border-radius:14px; padding:1.2rem; text-align:center;}
.pred h2 {margin:0; color:#166534; text-transform:capitalize;}
@media (prefers-color-scheme: dark){ .pred{background:#052e16;border-color:#166534;} .pred h2{color:#86efac;} }
</style>
<div class="hero"><h1>📰 Klasifikasi Berita Indonesia</h1>
<p>Crawling → Preprocessing → Word2Vec Skip-gram → Naive Bayes</p></div>
""", unsafe_allow_html=True)

# ---------- state ----------
ss = st.session_state
if "raw" not in ss:
    ss.raw = pd.read_csv(RAW_CSV) if RAW_CSV.exists() else None
if "clean" not in ss:
    ss.clean = pd.read_csv(CLEAN_CSV).fillna("") if CLEAN_CSV.exists() else None
if "result" not in ss:
    ss.result = None
if "models" not in ss:
    ss.models = md.load_models()
if "pp_opts" not in ss:
    ss.pp_opts = dict(use_stopwords=True, use_stemming=False, min_len=3)

tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["🏠 Beranda", "1️⃣ Crawling", "2️⃣ Preprocessing", "3️⃣ Modeling", "4️⃣ Klasifikasi"])

# ================= BERANDA =================
with tab0:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data mentah", len(ss.raw) if ss.raw is not None else 0)
    c2.metric("Data bersih", len(ss.clean) if ss.clean is not None else 0)
    c3.metric("Akurasi model", f"{ss.result['accuracy']:.1%}" if ss.result else "-")
    c4.metric("Model tersimpan", "Ya" if ss.models else "Belum")
    st.markdown("""
**Alur kerja**
1. **Crawling** – ambil URL dari RSS portal berita, lalu scrape isi artikelnya.
2. **Preprocessing** – case folding, cleaning, tokenizing, stopword removal, stemming (Sastrawi).
3. **Modeling** – latih **Word2Vec Skip-gram** (`sg=1`); vektor dokumen = rata-rata vektor kata.
4. **Klasifikasi** – **Gaussian Naive Bayes** memprediksi kategori berita baru (teks atau URL).

> GaussianNB dipakai karena embedding bernilai negatif, sehingga MultinomialNB tidak bisa dipakai.
""")

# ================= CRAWLING =================
with tab1:
    st.subheader("Crawling Berita detik.com")
    st.caption("Tempel link per kategori (satu link per baris). Boleh berupa halaman indeks/kategori "
               "(artikelnya dikumpulkan otomatis) atau link artikel langsung. "
               "Gunakan `{page}` pada link indeks untuk pagination.")
    with st.expander("🔗 Link sumber per kategori", expanded=True):
        sources = {}
        for cat, urls in crawler.DEFAULT_SOURCES.items():
            txt = st.text_area(f"Kategori: {cat}", "\n".join(urls), key=f"src_{cat}", height=90)
            sources[cat] = [u.strip() for u in txt.splitlines() if u.strip()]
        new_cat = st.text_input("Tambah kategori baru (opsional)", placeholder="mis. otomotif")
        if new_cat.strip():
            txt = st.text_area(f"Kategori: {new_cat}", key="src_new", height=90,
                               placeholder="https://oto.detik.com/indeks?page={page}")
            sources[new_cat.strip().lower()] = [u.strip() for u in txt.splitlines() if u.strip()]
    c1, c2, c3 = st.columns(3)
    per_cat = c1.slider("Artikel per kategori", 10, 200, 40, 10)
    max_pages = c2.slider("Halaman indeks maks.", 1, 20, 5)
    delay = c3.slider("Jeda antar request (detik)", 0.0, 3.0, 0.5, 0.1)
    append = st.checkbox("Gabungkan dengan data lama", True)

    with st.expander("🧪 Tes satu link artikel"):
        test_url = st.text_input("Link artikel detik.com", key="test_url")
        if test_url and st.button("Tes ambil artikel"):
            try:
                art = crawler.fetch_article(test_url)
                st.write("**Judul:**", art["title"])
                st.write("**Panjang isi:**", len(art["content"]), "karakter")
                st.text_area("Cuplikan isi", art["content"][:800], height=150, disabled=True)
            except Exception as e:
                st.error(f"Gagal: {e}")

    if st.button("🚀 Mulai Crawling", type="primary"):
        bar, msg = st.progress(0.0), st.empty()
        df = crawler.crawl({k: v for k, v in sources.items() if v}, per_cat, delay, max_pages,
                           progress=lambda f, m: (bar.progress(min(f, 1.0)), msg.caption(m)))
        if df.empty:
            st.error("Tidak ada artikel yang berhasil diambil. Cek koneksi internet / link sumber "
                     "(coba tes satu link artikel di atas).")
        else:
            if append and ss.raw is not None:
                df = pd.concat([ss.raw, df]).drop_duplicates("url").reset_index(drop=True)
            df.to_csv(RAW_CSV, index=False)
            ss.raw, ss.clean = df, None
            st.success(f"Selesai: {len(df)} artikel tersimpan.")

    st.markdown("**Atau unggah dataset sendiri** (CSV dengan kolom `content` dan `category`, opsional `title`)")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None and st.button("Gunakan dataset ini"):
        df = pd.read_csv(up)
        if {"content", "category"} <= set(df.columns):
            if "title" not in df.columns:
                df["title"] = ""
            df.to_csv(RAW_CSV, index=False)
            ss.raw, ss.clean = df, None
            st.success(f"{len(df)} baris dimuat.")
        else:
            st.error("CSV harus punya kolom `content` dan `category`.")

    if ss.raw is not None:
        st.markdown("### Data hasil crawling")
        cnt = ss.raw["category"].value_counts().reset_index()
        cnt.columns = ["kategori", "jumlah"]
        a, b = st.columns([1, 2])
        a.plotly_chart(px.bar(cnt, x="kategori", y="jumlah", color="kategori", height=300)
                       .update_layout(showlegend=False, margin=dict(t=10, b=10)), width="stretch")
        b.dataframe(ss.raw[["title", "category", "source", "content"]], height=300, width="stretch")
        st.download_button("⬇️ Unduh CSV", ss.raw.to_csv(index=False), "berita_raw.csv")

# ================= PREPROCESSING =================
with tab2:
    st.subheader("Preprocessing Teks")
    if ss.raw is None:
        st.info("Lakukan crawling atau unggah dataset terlebih dahulu.")
    else:
        if not pp.has_sastrawi():
            st.warning("PySastrawi belum terpasang: stopword & stemming memakai daftar minimal. "
                       "Jalankan `pip install PySastrawi`.")
        c1, c2, c3 = st.columns(3)
        use_sw = c1.checkbox("Hapus stopword", ss.pp_opts["use_stopwords"])
        use_st = c2.checkbox("Stemming (lambat, tapi lebih ringkas)", ss.pp_opts["use_stemming"])
        min_len = c3.slider("Panjang token minimum", 2, 5, ss.pp_opts["min_len"])
        opts = dict(use_stopwords=use_sw, use_stemming=use_st, min_len=min_len)

        st.markdown("**Demo langkah demi langkah**")
        i = st.number_input("Indeks artikel", 0, len(ss.raw) - 1, 0)
        steps = pp.preprocess_steps(str(ss.raw.iloc[int(i)]["content"]), **opts)
        for name, val in steps.items():
            if name == "tokens":
                continue
            show = " ".join(val[:60]) if isinstance(val, list) else val[:400]
            st.text_area(name, show, height=80, disabled=True, key=f"demo_{name}")

        if st.button("⚙️ Proses semua artikel", type="primary"):
            bar = st.progress(0.0)
            toks = []
            n = len(ss.raw)
            for k, txt in enumerate(ss.raw["content"].astype(str)):
                toks.append(pp.preprocess(txt, **opts))
                if k % 10 == 0:
                    bar.progress((k + 1) / n)
            bar.progress(1.0)
            clean = ss.raw.copy()
            clean["text_clean"] = [" ".join(t) for t in toks]
            clean = clean[clean["text_clean"].str.split().str.len() >= 5].reset_index(drop=True)
            clean.to_csv(CLEAN_CSV, index=False)
            ss.clean, ss.pp_opts = clean, opts
            st.success(f"{len(clean)} dokumen siap dimodelkan.")

        if ss.clean is not None:
            df = ss.clean
            all_tokens = [t for s in df["text_clean"] for t in s.split()]
            m1, m2, m3 = st.columns(3)
            m1.metric("Dokumen", len(df))
            m2.metric("Total token", f"{len(all_tokens):,}")
            m3.metric("Kosakata unik", f"{len(set(all_tokens)):,}")
            cat = st.selectbox("Kata terbanyak untuk kategori", sorted(df["category"].unique()))
            top = Counter(t for s in df[df.category == cat]["text_clean"] for t in s.split()).most_common(20)
            tdf = pd.DataFrame(top, columns=["kata", "frekuensi"])
            st.plotly_chart(px.bar(tdf, x="frekuensi", y="kata", orientation="h", height=450)
                            .update_layout(yaxis=dict(autorange="reversed")), width="stretch")
            st.dataframe(df[["category", "text_clean"]].head(50), width="stretch")

# ================= MODELING =================
with tab3:
    st.subheader("Skip-gram + Naive Bayes")
    if ss.clean is None:
        st.info("Selesaikan preprocessing dulu.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        vs = c1.select_slider("Dimensi vektor", [50, 100, 150, 200, 300], 100)
        win = c2.slider("Window", 2, 10, 5)
        mc = c3.slider("Min count", 1, 10, 2)
        ep = c4.slider("Epoch", 5, 100, 30, 5)
        d1, d2 = st.columns(2)
        ts = d1.slider("Porsi data uji", 0.1, 0.5, 0.2, 0.05)
        vsm = d2.select_slider("var_smoothing (GaussianNB)", [1e-9, 1e-7, 1e-5, 1e-3, 1e-1], 1e-9,
                               format_func=lambda x: f"{x:.0e}")

        if st.button("🧠 Latih Model", type="primary"):
            vc = ss.clean["category"].value_counts()
            if vc.min() < 2 or ss.clean["category"].nunique() < 2:
                st.error("Setiap kategori minimal 2 dokumen dan minimal ada 2 kategori.")
            else:
                with st.spinner("Melatih Skip-gram & Naive Bayes..."):
                    corpus = [s.split() for s in ss.clean["text_clean"]]
                    ss.result = md.train_all(corpus, ss.clean["category"].values, vs, win, mc, ep, ts, vsm)
                    ss.models = md.load_models()
                st.success("Model selesai dilatih & disimpan di folder `models/`.")

        r = ss.result
        if r:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Akurasi", f"{r['accuracy']:.2%}")
            m2.metric("Macro F1", f"{r['report']['macro avg']['f1-score']:.2%}")
            m3.metric("Data latih / uji", f"{r['n_train']} / {r['n_test']}")
            m4.metric("Kosakata Skip-gram", f"{r['vocab']:,}")

            a, b = st.columns(2)
            with a:
                st.markdown("**Confusion matrix**")
                fig = px.imshow(r["cm"], x=r["classes"], y=r["classes"], text_auto=True,
                                color_continuous_scale="Blues", labels=dict(x="Prediksi", y="Aktual"))
                st.plotly_chart(fig, width="stretch")
            with b:
                st.markdown("**Classification report**")
                rep = pd.DataFrame(r["report"]).T.drop(columns="support", errors="ignore").round(3)
                st.dataframe(rep, width="stretch", height=380)

            st.markdown("### Eksplorasi embedding Skip-gram")
            wv = r["w2v"].wv
            q = st.text_input("Cari kata yang mirip", wv.index_to_key[0] if len(wv) else "")
            if q in wv.key_to_index:
                sim = pd.DataFrame(wv.most_similar(q, topn=10), columns=["kata", "kemiripan"])
                st.dataframe(sim, width="stretch")
            elif q:
                st.warning("Kata tidak ada di kosakata model.")
            n_words = st.slider("Jumlah kata pada plot PCA", 30, 200, 80, 10)
            words = wv.index_to_key[:n_words]
            if len(words) >= 3:
                xy = PCA(n_components=2).fit_transform(np.array([wv[w] for w in words]))
                pdf = pd.DataFrame(xy, columns=["x", "y"]); pdf["kata"] = words
                st.plotly_chart(px.scatter(pdf, x="x", y="y", text="kata", height=520)
                                .update_traces(textposition="top center"), width="stretch")

# ================= KLASIFIKASI =================
with tab4:
    st.subheader("Klasifikasi Berita Baru")
    models = ss.models
    if not models:
        st.info("Latih model terlebih dahulu di tab Modeling.")
    else:
        mode = st.radio("Sumber input", ["Tempel teks", "URL berita"], horizontal=True)
        text = ""
        if mode == "Tempel teks":
            text = st.text_area("Isi berita", height=220)
        else:
            url = st.text_input("URL artikel")
            if url and st.button("Ambil artikel"):
                try:
                    ss.fetched = crawler.extract_article(url)
                except Exception as e:
                    st.error(f"Gagal mengambil artikel: {e}")
            text = ss.get("fetched", "")
            if text:
                st.text_area("Teks hasil crawling", text, height=180, disabled=True)

        if st.button("🔍 Klasifikasikan", type="primary") and text.strip():
            tokens = pp.preprocess(text, **ss.pp_opts)
            proba, known = md.predict_proba(tokens, models["w2v"], models["clf"])
            best = max(proba, key=proba.get)
            if not known:
                st.warning("Tidak ada kata yang dikenal model; hasil tidak dapat diandalkan.")
            st.markdown(f"<div class='pred'>Prediksi kategori<h2>{best}</h2>"
                        f"Keyakinan {proba[best]:.1%}</div>", unsafe_allow_html=True)
            pdf = pd.DataFrame({"kategori": list(proba), "probabilitas": list(proba.values())})
            st.plotly_chart(px.bar(pdf.sort_values("probabilitas"), x="probabilitas", y="kategori",
                                   orientation="h", range_x=[0, 1], height=300), width="stretch")
            st.caption(f"{len(known)} dari {len(tokens)} token dikenal oleh model Skip-gram.")
