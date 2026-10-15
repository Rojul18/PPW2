"""Tahap 3 & 4 - Word2Vec Skip-gram + klasifikasi Naive Bayes."""
from pathlib import Path
import joblib
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def train_skipgram(corpus, vector_size=100, window=5, min_count=2, epochs=30, seed=42):
    """Word2Vec dengan sg=1 (Skip-gram)."""
    return Word2Vec(sentences=corpus, vector_size=vector_size, window=window,
                    min_count=min_count, sg=1, epochs=epochs, workers=1, seed=seed)


def doc_vector(tokens, wv):
    """Vektor dokumen = rata-rata vektor kata Skip-gram."""
    vecs = [wv[t] for t in tokens if t in wv.key_to_index]
    if not vecs:
        return np.zeros(wv.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)


def vectorize(corpus, wv):
    return np.vstack([doc_vector(t, wv) for t in corpus])


def train_all(corpus, labels, vector_size=100, window=5, min_count=2, epochs=30,
              test_size=0.2, var_smoothing=1e-9, seed=42):
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    tr, te = train_test_split(idx, test_size=test_size, stratify=labels, random_state=seed)
    train_corpus = [corpus[i] for i in tr]
    test_corpus = [corpus[i] for i in te]

    # Skip-gram hanya dilatih dari data latih -> tidak ada kebocoran data uji.
    w2v = train_skipgram(train_corpus, vector_size, window, min_count, epochs, seed)
    X_tr, X_te = vectorize(train_corpus, w2v.wv), vectorize(test_corpus, w2v.wv)
    y_tr, y_te = labels[tr], labels[te]

    # GaussianNB dipilih karena vektor embedding bernilai negatif (MultinomialNB tidak bisa).
    clf = GaussianNB(var_smoothing=var_smoothing).fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    classes = list(clf.classes_)

    result = {
        "w2v": w2v, "clf": clf, "classes": classes,
        "accuracy": accuracy_score(y_te, y_pred),
        "report": classification_report(y_te, y_pred, output_dict=True, zero_division=0),
        "cm": confusion_matrix(y_te, y_pred, labels=classes),
        "n_train": len(tr), "n_test": len(te), "vocab": len(w2v.wv),
        "y_test": y_te, "y_pred": y_pred, "test_idx": te,
    }
    w2v.save(str(MODEL_DIR / "skipgram.model"))
    joblib.dump({"clf": clf, "classes": classes}, MODEL_DIR / "naive_bayes.joblib")
    return result


def load_models():
    p1, p2 = MODEL_DIR / "skipgram.model", MODEL_DIR / "naive_bayes.joblib"
    if not (p1.exists() and p2.exists()):
        return None
    return {"w2v": Word2Vec.load(str(p1)), **joblib.load(p2)}


def predict_proba(tokens, w2v, clf):
    vec = doc_vector(tokens, w2v.wv).reshape(1, -1)
    proba = clf.predict_proba(vec)[0]
    known = [t for t in tokens if t in w2v.wv.key_to_index]
    return dict(zip(clf.classes_, proba)), known
