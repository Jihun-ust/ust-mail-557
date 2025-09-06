
import re, math, json, random
import numpy as np, pandas as pd
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

#  Text utils 
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def sent_tokenize(text: str) -> List[str]:
    # Lightweight sentence split
    parts = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    return [p for p in parts if p]

def approx_tokens(text: str) -> int:
    return max(1, len(text.split()))

#  Chunkers 
def chunk_by_words(text: str, max_words: int = 180, overlap: int = 30) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_words]
        chunks.append(" ".join(chunk))
        i += max_words - overlap
        if i <= 0: break
    return chunks

def chunk_recursive_by_headers(text: str, header_pattern=r"^#+\s+.*$", max_words: int = 220) -> List[str]:
    # Split by markdown headers, then further split long blocks
    blocks = re.split(header_pattern, text, flags=re.MULTILINE)
    chunks = []
    for b in blocks:
        b = normalize_ws(b)
        if not b: continue
        if approx_tokens(b) <= max_words:
            chunks.append(b)
        else:
            chunks.extend(chunk_by_words(b, max_words=max_words, overlap=40))
    return chunks

def sliding_window(sentences: List[str], window: int = 4, stride: int = 2) -> List[str]:
    chunks = []
    for i in range(0, len(sentences), stride):
        w = sentences[i:i+window]
        if not w: break
        chunks.append(" ".join(w))
        if i+window >= len(sentences): break
    return chunks

#  Embeddings (TF-IDF as dense stand-in) 
class TFIDFIndex:
    def __init__(self, docs: List[str]):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=1)
        self.X = self.vectorizer.fit_transform(docs)
        self.docs = docs
        self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn.fit(self.X)

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        q = self.vectorizer.transform([query])
        dist, idx = self.nn.kneighbors(q, n_neighbors=min(k, self.X.shape[0]))
        # cosine distance -> similarity
        sims = 1 - dist[0]
        return list(zip(idx[0].tolist(), sims.tolist()))

    def embed(self, texts: List[str]):
        return self.vectorizer.transform(texts)

#  Simple BM25 (Okapi) 
class BM25Okapi:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1, self.b = k1, b
        self.N = len(corpus)
        self.df = {}
        self.avgdl = np.mean([len(doc) for doc in corpus]) if self.N else 0.0
        for doc in corpus:
            seen = set()
            for w in doc:
                if w not in seen:
                    self.df[w] = self.df.get(w, 0) + 1
                    seen.add(w)

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0) + 1e-9
        return math.log(1 + (self.N - n + 0.5)/(n + 0.5))

    def score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        score = 0.0
        dl = len(doc_tokens) + 1e-9
        tf = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1
        for q in query_tokens:
            f = tf.get(q, 0)
            idf = self.idf(q)
            denom = f + self.k1*(1 - self.b + self.b * dl / (self.avgdl + 1e-9))
            score += idf * ((f*(self.k1+1)) / (denom + 1e-9))
        return score

    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        q = query.lower().split()
        scores = []
        for i, doc in enumerate(self.corpus):
            s = self.score(q, doc)
            scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

#  MMR (diversify) 
def mmr(query_vec, doc_vecs, doc_ids, lambda_=0.7, top_k=5):
    selected = []
    candidates = set(range(len(doc_ids)))
    # Precompute sim to query
    sims = cosine_similarity(query_vec, doc_vecs).flatten()
    while len(selected) < min(top_k, len(doc_ids)):
        mmr_scores = []
        for i in candidates:
            if not selected:
                div = 0
            else:
                div = max(cosine_similarity(doc_vecs[i], doc_vecs[selected]).flatten())
            score = lambda_*sims[i] - (1-lambda_)*div
            mmr_scores.append((score, i))
        mmr_scores.sort(reverse=True)
        best = mmr_scores[0][1]
        selected.append(best)
        candidates.remove(best)
    return [doc_ids[i] for i in selected]

#  RAG pipeline helpers 
def build_chunks(df: pd.DataFrame, strategy="recursive_headers"):
    chunks = []
    mapping = []
    for i, row in df.iterrows():
        text = row["text"]
        if strategy == "recursive_headers":
            cs = chunk_recursive_by_headers(text, max_words=200)
        elif strategy == "words":
            cs = chunk_by_words(text, max_words=160, overlap=20)
        else:
            sents = sent_tokenize(text)
            cs = sliding_window(sents, window=4, stride=2)
        for c in cs:
            chunks.append(c)
            mapping.append((row["doc_id"], row["title"]))
    return chunks, mapping

def hybrid_search(query, tfidf_idx: TFIDFIndex, bm25: BM25Okapi, chunks: List[str], k=5):
    # Normalize scores to 0-1 and combine
    tf_hits = tfidf_idx.search(query, k=k*5)
    bm_hits = bm25.search(query, k=k*5)
    def norm(scores):
        vals = np.array([s for _, s in scores])
        if len(vals)==0: return {}
        mi, ma = vals.min(), vals.max()
        rng = (ma-mi) if ma>mi else 1.0
        return {i: (s-mi)/rng for i, s in scores}
    tfd = norm(tf_hits); bmd = norm(bm_hits)
    ids = set([i for i,_ in tf_hits] + [i for i,_ in bm_hits])
    combined = [(i, 0.6*tfd.get(i,0) + 0.4*bmd.get(i,0)) for i in ids]
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:k]

# "LLM" simulator for teaching (rule-based templater)
def llm_simulate(prompt: str, retrieved: List[str] = None, chain_of_thought: bool = False) -> str:
    # This is a pedagogical stub; it reads the prompt and retrieved context, then formats a helpful answer.
    context = " ".join(retrieved[:2]) if retrieved else ""
    # Simple intent rules
    if "warranty" in prompt.lower():
        answer = "Standard warranty: 12 months (small) / 24 months (major); claims via support portal with proof of purchase."
    elif "return" in prompt.lower():
        answer = "Returns: 30 days; 10% restock fee for opened items unless defective; holidays (Nov 15–Dec 31) allow 60 days."
    elif "install" in prompt.lower() or "installation" in prompt.lower():
        answer = "Install requires 120V, water line, level surface; test with an empty cycle; fix leaks by tightening clamps and checking gaskets."
    else:
        answer = "Based on our docs: " + (context[:220] + ("..." if len(context)>220 else ""))
    if chain_of_thought:
        rationale = "I looked for exact policy statements and time windows; I prioritized official policy docs; I then summarized and listed steps."
        return f"Reasoning: {rationale}\n\nAnswer: {answer}"
    return answer

# Tool registry & function-calling simulation
def tool_track_order(order_id: str) -> dict:
    return {"tool":"track_order","order_id":order_id,"status":"in_transit","eta_days":3}

def tool_schedule_tech(model: str, zip_code: str) -> dict:
    return {"tool":"schedule_technician","model":model,"zip":zip_code,"first_available_window":"Tue 1-5pm"}

def tool_start_rma(serial: str, reason: str) -> dict:
    rid = "RMA-" + serial[-4:] + "-" + str(random.randint(100,999))
    return {"tool":"start_rma","serial":serial,"reason":reason,"ticket_id":rid}

TOOLS = {
    "track_order": tool_track_order,
    "schedule_tech": tool_schedule_tech,
    "start_rma": tool_start_rma,
}

def simple_router(user_msg: str) -> Tuple[str, Dict]:
    s = user_msg.lower()
    if "where is my order" in s or "track" in s:
        return "track_order", {"order_id":"NW123456789"}
    if "technician" in s or "install" in s or "service" in s:
        return "schedule_tech", {"model":"Model B","zip_code":"60601"}
    if "warranty" in s and ("claim" in s or "rma" in s or "return" in s):
        return "start_rma", {"serial":"SN-A1B2C3D4","reason":"defect: leak persists after install checks"}
    return "none", {}

#  JSON validation (light) 
def validate_json(payload: dict, required: List[str]) -> Tuple[bool, List[str]]:
    missing = [k for k in required if k not in payload]
    return (len(missing)==0, missing)
