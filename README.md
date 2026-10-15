# Klasifikasi Berita - UTS PPW
Crawling detik.com -> Preprocessing -> Word2Vec Skip-gram -> Gaussian Naive Bayes (Streamlit).

## Menjalankan
```
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Urutan pakai: tab Crawling detik.com -> Preprocessing -> Modeling -> Klasifikasi.
Data tersimpan di `data/`, model di `models/`.
