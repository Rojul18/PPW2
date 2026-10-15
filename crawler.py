"""Tahap 1 - Crawling berita detik.com berbasis link.

Input per kategori berupa daftar link:
  * link halaman indeks / kategori  -> crawler mengumpulkan link artikel di dalamnya
    (boleh memakai placeholder {page} untuk pagination, mis. .../indeks?page={page})
  * link artikel langsung (berpola /d-<angka>/) -> langsung diambil
"""
import re
import time
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
}

# Link awal per kategori (bisa diedit di aplikasi). Cek di browser bila hasil kosong.
DEFAULT_SOURCES = {
    "berita":    ["https://news.detik.com/indeks?page={page}", "https://news.detik.com/"],
    "ekonomi":   ["https://finance.detik.com/indeks?page={page}", "https://finance.detik.com/"],
    "olahraga":  ["https://sport.detik.com/indeks?page={page}", "https://sport.detik.com/"],
    "teknologi": ["https://inet.detik.com/indeks?page={page}", "https://inet.detik.com/"],
    "hiburan":   ["https://hot.detik.com/indeks?page={page}", "https://hot.detik.com/"],
}

_ARTICLE_RE = re.compile(r"/d-\d+")
_JUNK_PREFIX = ("baca juga", "simak video", "saksikan", "video:", "lihat juga", "baca selengkapnya",
                "ikuti kami", "download", "dapatkan", "gabung")


def is_article_url(url: str) -> bool:
    return bool(_ARTICLE_RE.search(urlparse(url).path))


def _get(url: str, timeout: int = 15) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r


def extract_links(html: str, base_url: str) -> list:
    """Ambil link artikel detik.com (berpola /d-<id>/) dari sebuah halaman."""
    soup = BeautifulSoup(html, "html.parser")
    out, seen = [], set()
    for a in soup.find_all("a", href=True):
        url = urljoin(base_url, a["href"]).split("#")[0].split("?")[0]
        host = urlparse(url).netloc
        if host.endswith("detik.com") and is_article_url(url) and url not in seen:
            seen.add(url)
            out.append(url)
    return out


def collect_links(index_url: str, max_pages: int = 5, delay: float = 0.5) -> list:
    """Kumpulkan link artikel dari halaman indeks (dengan pagination bila ada {page})."""
    if "{page}" in index_url:
        pages = [index_url.replace("{page}", str(p)) for p in range(1, max_pages + 1)]
    else:
        pages = [index_url]
    links, seen = [], set()
    for pu in pages:
        try:
            found = extract_links(_get(pu).text, pu)
        except Exception:
            break
        new = [u for u in found if u not in seen]
        if not new:          # halaman tidak memberi link baru -> berhenti
            break
        seen.update(new)
        links += new
        time.sleep(delay)
    return links


def _normalize(text: str) -> str:
    text = re.sub(r"\(\s*[a-z]{2,5}\s*/\s*[a-z]{2,5}\s*\)\s*$", "", text.strip(), flags=re.I)  # inisial penulis
    text = re.sub(r"^\s*[A-Z][\w\s,.]{1,30}\s[-–]\s", "", text)                              # "Jakarta - "
    return re.sub(r"\s+", " ", text).strip()


def fetch_article(url: str) -> dict:
    """Ambil judul, tanggal, dan isi artikel detik.com."""
    # ?single=1 meminta artikel ditampilkan dalam satu halaman penuh
    try:
        html = _get(url + ("&" if "?" in url else "?") + "single=1").text
    except Exception:
        html = _get(url).text
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.select_one("h1.detail__title") or soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""
    if not title and soup.find("meta", property="og:title"):
        title = soup.find("meta", property="og:title").get("content", "")
    pub = soup.find("meta", attrs={"name": "publishdate"})
    published = pub.get("content", "") if pub else ""

    body = (soup.select_one("div.detail__body-text") or soup.select_one("div.itp_bodycontent")
            or soup.find("article") or soup)
    for t in body.select("script, style, noscript, table, .staticdetail_container, .detail__body-tag, "
                         ".parallaxindetail, .para_caption, .sisip_embed_sosmed, .lihatjg, .linksisip, "
                         ".detail__body-tag, figure, iframe"):
        t.decompose()
    paras = [p.get_text(" ", strip=True) for p in body.find_all("p")]
    paras = [p for p in paras if len(p) > 30 and not p.lower().startswith(_JUNK_PREFIX)]
    return {"title": title, "published": published, "content": _normalize(" ".join(paras))}


def extract_article(url: str) -> str:
    """Hanya teks isi artikel (dipakai tab Klasifikasi untuk input URL)."""
    return fetch_article(url)["content"]


def crawl(sources: dict, per_category: int = 40, delay: float = 0.5, max_pages: int = 5,
          progress=None) -> pd.DataFrame:
    rows, seen = [], set()
    cats = list(sources.items())
    for ci, (cat, links) in enumerate(cats):
        # 1) kumpulkan semua URL artikel kandidat untuk kategori ini
        candidates = []
        for link in links:
            if is_article_url(link):
                candidates.append(link)
            else:
                candidates += collect_links(link, max_pages, delay)
        candidates = [u for u in dict.fromkeys(candidates) if u not in seen]

        # 2) ambil isi tiap artikel sampai target tercapai
        got = 0
        for url in candidates:
            if got >= per_category:
                break
            seen.add(url)
            try:
                art = fetch_article(url)
                time.sleep(delay)
            except Exception:
                continue
            if len(art["content"]) < 200:
                continue
            rows.append({"title": art["title"], "url": url, "category": cat,
                         "source": urlparse(url).netloc, "published": art["published"],
                         "content": art["content"]})
            got += 1
            if progress:
                progress((ci + got / max(per_category, 1)) / len(cats),
                         f"[{cat}] {got}/{per_category} - {art['title'][:60]}")
    return pd.DataFrame(rows)
