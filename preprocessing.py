"""Tahap 2 - Preprocessing teks berbahasa Indonesia."""
import re
from functools import lru_cache

try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _HAS_SASTRAWI = True
except Exception:  # pragma: no cover
    _HAS_SASTRAWI = False

# Kata "bocor" khas media yang bisa membuat model menebak sumber, bukan topik.
EXTRA_STOPWORDS = {
    "antara", "cnn", "detikcom", "detiknews", "detikfinance", "detiksport", "detikinet", "detikhot", "dtk", "sumber", "indonesia", "jakarta", "kompas", "detik", "com", "foto", "video",
    "baca", "juga", "klik", "simak", "red", "ed", "pewarta", "editor", "copyright",
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk", "pada", "adalah",
    "dalam", "akan", "oleh", "atau", "sebagai", "karena", "juga", "telah", "sudah", "bisa",
}

_FALLBACK_STOP = EXTRA_STOPWORDS


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset:
    words = set(EXTRA_STOPWORDS)
    if _HAS_SASTRAWI:
        words |= set(StopWordRemoverFactory().get_stop_words())
    return frozenset(words)


@lru_cache(maxsize=1)
def _stemmer():
    return StemmerFactory().create_stemmer() if _HAS_SASTRAWI else None


@lru_cache(maxsize=200000)
def _stem_word(w: str) -> str:
    st = _stemmer()
    return st.stem(w) if st else w


def case_folding(text: str) -> str:
    return str(text).lower()


def cleaning(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URL
    text = re.sub(r"<[^>]+>", " ", text)                  # sisa tag HTML
    text = re.sub(r"[^a-z\s]", " ", text)                 # hanya huruf
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list:
    return text.split()


def remove_stopwords(tokens: list, min_len: int = 3) -> list:
    sw = get_stopwords()
    return [t for t in tokens if t not in sw and len(t) >= min_len]


def stem(tokens: list) -> list:
    return [_stem_word(t) for t in tokens]


def preprocess_steps(text: str, use_stopwords=True, use_stemming=False, min_len=3) -> dict:
    """Jalankan pipeline dan kembalikan hasil tiap langkah (untuk demo di UI)."""
    s1 = case_folding(text)
    s2 = cleaning(s1)
    s3 = tokenize(s2)
    s4 = remove_stopwords(s3, min_len) if use_stopwords else [t for t in s3 if len(t) >= min_len]
    s5 = stem(s4) if use_stemming else s4
    if use_stemming:  # stemming bisa menghasilkan stopword/kata pendek baru
        s5 = [t for t in s5 if len(t) >= min_len and (not use_stopwords or t not in get_stopwords())]
    return {"Case folding": s1, "Cleaning": s2, "Tokenizing": s3,
            "Stopword removal": s4, "Stemming": s5, "tokens": s5}


def preprocess(text: str, **kw) -> list:
    return preprocess_steps(text, **kw)["tokens"]


def has_sastrawi() -> bool:
    return _HAS_SASTRAWI
