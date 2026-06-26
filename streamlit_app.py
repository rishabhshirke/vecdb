import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vectordb import VectorDB

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VecDB Studio",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0b0d14; }
[data-testid="stSidebar"] { background: #0f111a; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #161825;
    border-radius: 6px;
    padding: 8px 18px;
    color: #94a3b8;
    border: 1px solid #1e2235;
}
.stTabs [aria-selected="true"] {
    background: #1e1b4b;
    color: #a78bfa;
    border-color: #4c1d95;
}
.chunk-box {
    background: #111827;
    border-left: 3px solid #6d28d9;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    color: #e2e8f0;
}
.step-header {
    background: linear-gradient(90deg, #1e1b4b, #0b0d14);
    border-left: 3px solid #7c3aed;
    border-radius: 4px;
    padding: 10px 16px;
    margin: 16px 0 8px 0;
    color: #c4b5fd;
    font-weight: 600;
    font-size: 15px;
}
.metric-pill {
    display: inline-block;
    background: #1e2235;
    border: 1px solid #313244;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    color: #94a3b8;
    margin: 3px;
}
</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "db": None,
    "chunks": [],
    "vectors": [],
    "collection_name": None,
    "processed": False,
    "raw_text": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.db is None:
    st.session_state.db = VectorDB()

# ─── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model (one-time download)...")
def load_model():
    # Try fastembed first (lightweight, ONNX-based, no torchvision needed)
    try:
        from fastembed import TextEmbedding
        model = TextEmbedding("BAAI/bge-small-en-v1.5")
        model._type = "fastembed"
        return model
    except ImportError:
        pass
    # Fall back to sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model._type = "sentence_transformers"
        return model
    except Exception:
        return None


def extract_text(file) -> str:
    if file.type == "text/plain":
        return file.read().decode("utf-8", errors="replace")
    elif file.type == "application/pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(file)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            st.error("Install pypdf: `pip install pypdf`")
            return ""
    return ""


def chunk_text(text: str, size: int, overlap: int) -> list:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "start": start, "end": end})
        if end == len(text):
            break
        start += size - overlap
    return chunks


def embed(text: str, model, dim: int) -> np.ndarray:
    if model is not None:
        try:
            if getattr(model, "_type", "") == "fastembed":
                return np.array(list(model.embed([text]))[0], dtype=np.float32)
            else:
                return model.encode(text, show_progress_bar=False).astype(np.float32)
        except Exception:
            pass
    rng = np.random.default_rng(abs(hash(text)) % 2**31)
    return rng.random(dim).astype(np.float32)


def pca_2d(vectors: list) -> np.ndarray:
    mat = np.array(vectors, dtype=np.float32)
    if mat.shape[0] < 2:
        return np.column_stack([np.zeros(mat.shape[0]), np.zeros(mat.shape[0])])
    try:
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(mat)
    except ImportError:
        # Manual 2-component PCA fallback
        mat -= mat.mean(axis=0)
        _, _, vt = np.linalg.svd(mat, full_matrices=False)
        return (mat @ vt[:2].T)


def scatter_fig(coords, chunks, highlight_last=False, height=380):
    n = len(coords)
    colors = [f"hsl({int(220 + 80 * i / max(n-1, 1))}, 70%, 60%)" for i in range(n)]
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=coords[:, 0], y=coords[:, 1],
        mode="markers+text",
        marker=dict(size=13, color=list(range(n)), colorscale="Viridis",
                    showscale=True, colorbar=dict(title="Chunk #", thickness=12),
                    line=dict(color="white", width=1)),
        text=[f"C{i}" for i in range(n)],
        textposition="top center",
        textfont=dict(color="white", size=10),
        hovertext=[f"Chunk {i}<br><br>{chunks[i]['text'][:120]}..." for i in range(n)],
        hoverinfo="text",
        name="vectors",
    ))

    if highlight_last and n > 0:
        fig.add_trace(go.Scatter(
            x=[coords[-1, 0]], y=[coords[-1, 1]],
            mode="markers",
            marker=dict(size=22, color="#fbbf24", symbol="star",
                        line=dict(color="white", width=2)),
            hoverinfo="skip", name="new",
        ))

    fig.update_layout(
        height=height,
        paper_bgcolor="#0b0d14", plot_bgcolor="#111827",
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(title="PC1", color="#6b7280", gridcolor="#1e2235", zeroline=False),
        yaxis=dict(title="PC2", color="#6b7280", gridcolor="#1e2235", zeroline=False),
        showlegend=False, font=dict(color="#e2e8f0"),
    )
    return fig


def vec_bar_fig(vec: np.ndarray, n=64, height=110):
    v = vec[:n]
    fig = go.Figure(go.Bar(
        x=list(range(len(v))), y=v.tolist(),
        marker_color=["#7c3aed" if x >= 0 else "#ef4444" for x in v],
    ))
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showgrid=False, color="#4b5563"),
        showlegend=False,
    )
    return fig


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    chunk_size = st.slider("Chunk size (chars)", 100, 1200, 400, 50)
    overlap    = st.slider("Overlap (chars)", 0, 200, 60, 10)
    anim_speed = st.slider("Animation speed", 0.05, 1.5, 0.35, 0.05,
                           help="Seconds per step — lower = faster")
    st.divider()
    index_type = st.radio("Index", ["hnsw", "flat"], horizontal=True)
    metric     = st.selectbox("Metric", ["cosine", "euclidean", "dot"])
    st.divider()
    st.markdown("**Embedding model**")
    use_model = st.toggle("Use sentence-transformers", value=True,
                          help="Requires: pip install sentence-transformers")
    if not use_model:
        st.caption("Random 128-dim vectors will be used (demo mode)")


# ─── Header ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown("# 🔮 VecDB Studio")
    st.caption("Upload a document → watch it chunk → get embedded → flow into vector space")
with c2:
    if st.session_state.processed:
        n_vecs = len(st.session_state.vectors)
        dim_val = len(st.session_state.vectors[0]) if n_vecs else 0
        st.metric("Vectors stored", n_vecs)
        st.caption(f"dim = {dim_val}")

st.divider()

# ─── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a document (TXT or PDF)",
    type=["txt", "pdf"],
    label_visibility="collapsed",
)

# Clear everything when file is removed
if uploaded_file is None and st.session_state.processed:
    if st.session_state.collection_name:
        try:
            st.session_state.db.delete_collection(st.session_state.collection_name)
        except Exception:
            pass
    st.session_state.chunks = []
    st.session_state.vectors = []
    st.session_state.processed = False
    st.session_state.collection_name = None
    st.session_state.raw_text = ""
    st.rerun()

if uploaded_file:
    raw_text = extract_text(uploaded_file)
    st.session_state.raw_text = raw_text

    info_cols = st.columns(4)
    info_cols[0].metric("Characters", f"{len(raw_text):,}")
    info_cols[1].metric("Words", f"{len(raw_text.split()):,}")
    preview_chunks = chunk_text(raw_text, chunk_size, overlap)
    info_cols[2].metric("Estimated chunks", len(preview_chunks))
    info_cols[3].metric("Chunk size", chunk_size)

    run = st.button("🚀 Process Document", type="primary", use_container_width=True)

    if run:
        # ── Model + collection setup ────────────────────────────────────────────
        model = load_model() if use_model else None
        if model is None:
            dim = 128
        elif getattr(model, "_type", "") == "fastembed":
            dim = len(list(model.embed(["test"]))[0])
        else:
            dim = model.get_sentence_embedding_dimension()

        cname = f"doc_{uuid.uuid4().hex[:6]}"
        if st.session_state.collection_name:
            try:
                st.session_state.db.delete_collection(st.session_state.collection_name)
            except Exception:
                pass

        col_obj = st.session_state.db.create_collection(
            cname, dim=dim, metric=metric, index_type=index_type
        )
        st.session_state.collection_name = cname

        chunks = chunk_text(raw_text, chunk_size, overlap)
        all_vectors: list = []
        all_chunks:  list = []

        # ── Live pipeline tab only ──────────────────────────────────────────────
        tab_live, = st.tabs(["🎬 Live Pipeline"])

        # ══════════════════════════════════════════════════════════════
        # TAB 1 — Live Pipeline (runs during processing)
        # ══════════════════════════════════════════════════════════════
        with tab_live:

            COLORS = [
                "#7c3aed", "#0ea5e9", "#10b981", "#f59e0b",
                "#ef4444", "#ec4899", "#14b8a6", "#8b5cf6",
            ]

            # ── STEP 1: Chunking ───────────────────────────────────────
            st.markdown('<div class="step-header">📦 Step 1 — Chunking the Document</div>',
                        unsafe_allow_html=True)

            st.markdown("""
> **What is chunking?**
>
> Language models and embedding models have a maximum input length (called a *context window*).
> A large document — say 50 pages — can't be fed in all at once. So we split it into smaller
> overlapping pieces called **chunks**. Each chunk becomes one unit that will be embedded and stored.
""")

            with st.expander("📖 Why overlap? Why not just cut cleanly?"):
                st.markdown("""
**The problem with clean cuts:**
Imagine a sentence like *"The neural network failed because the learning rate was too high"*
cut right after *"failed"*. The first chunk ends with an incomplete thought.
The second chunk starts with *"because the learning rate"* — missing context.

**Overlap solves this:**
With an overlap of 60 characters, the end of chunk N is repeated at the start of chunk N+1.
This ensures no important idea gets split across two chunks and lost.

```
Chunk 1:  [......... text .........|-- overlap --|]
Chunk 2:              [-- overlap --|......... text .........|-- overlap --|]
Chunk 3:                                          [-- overlap --|.........]
```

**Rule of thumb:**
- Chunk size: 300–500 chars for precise retrieval, 800–1200 for more context per result
- Overlap: 10–20% of chunk size
""")

            chunk_prog  = st.progress(0)
            live_explain = st.empty()
            doc_display = st.empty()
            chunk_info  = st.empty()

            for i, chunk in enumerate(chunks):
                chunk_prog.progress((i + 1) / len(chunks),
                                    text=f"Chunking… {i+1}/{len(chunks)}")

                with live_explain.container():
                    st.info(
                        f"**Extracting chunk {i+1} of {len(chunks)}** — "
                        f"characters `{chunk['start']}` to `{chunk['end']}` "
                        f"({len(chunk['text'])} chars, {len(chunk['text'].split())} words). "
                        + (f"This chunk **overlaps {overlap} chars** with the previous one — "
                           f"those shared characters preserve context across the boundary."
                           if i > 0 else
                           "This is the **first chunk** — it starts at the beginning of the document.")
                    )

                # Build highlighted document view
                html = (
                    "<div style='font-family:\"Courier New\",monospace;font-size:12.5px;"
                    "line-height:1.7;background:#0d1117;padding:14px;border-radius:8px;"
                    "max-height:260px;overflow-y:auto;color:#cbd5e1'>"
                )
                cursor = 0
                for j, c in enumerate(chunks[:i + 1]):
                    if c["start"] > cursor:
                        gap = raw_text[cursor:c["start"]].replace("\n", "<br>").replace(" ", "&nbsp;")
                        html += f"<span style='color:#374151'>{gap}</span>"
                    color = COLORS[j % len(COLORS)]
                    snippet = raw_text[c["start"]:c["end"]].replace("\n", "<br>").replace(" ", "&nbsp;")
                    alpha = "44" if j < i else "88"
                    html += (
                        f"<span style='background:{color}{alpha};border-bottom:2px solid {color};"
                        f"border-radius:3px;padding:1px 0'>{snippet}</span>"
                    )
                    cursor = c["end"]
                if cursor < len(raw_text):
                    rest = raw_text[cursor:].replace("\n", "<br>").replace(" ", "&nbsp;")
                    html += f"<span style='color:#374151'>{rest}</span>"
                html += "</div>"

                doc_display.markdown(html, unsafe_allow_html=True)
                chunk_info.caption(
                    f"🟣 Chunk {i+1}: chars {chunk['start']}–{chunk['end']} "
                    f"| {len(chunk['text'])} chars | {len(chunk['text'].split())} words"
                )
                time.sleep(anim_speed * 0.4)

            chunk_prog.progress(1.0, text=f"✅ {len(chunks)} chunks extracted")
            live_explain.empty()

            st.success(
                f"✅ **Chunking complete.** Your document ({len(raw_text):,} chars) was split into "
                f"**{len(chunks)} chunks** of ~{chunk_size} chars each with {overlap}-char overlap. "
                f"Each chunk is now an independent unit ready to be embedded."
            )

            st.divider()

            # ── STEP 2: Embedding ──────────────────────────────────────
            st.markdown('<div class="step-header">🧠 Step 2 — Computing Embeddings</div>',
                        unsafe_allow_html=True)

            st.markdown("""
> **What is an embedding?**
>
> A computer can't understand text — it only understands numbers. An **embedding** is how we
> translate a piece of text into a list of numbers (a *vector*) that captures its *meaning*.
> The model (`all-MiniLM-L6-v2`) reads each chunk and outputs **{dim} numbers** — one for each
> learned dimension of semantic meaning.
""".format(dim=dim))

            with st.expander("📖 What do the numbers actually mean?"):
                st.markdown(f"""
**Each dimension = one learned feature of meaning.**

The model was trained on millions of sentences. Through training it learned to represent
concepts like *"this text is about technology"*, *"this has a negative sentiment"*,
*"this mentions time"* as specific patterns across the {dim} dimensions.

No single dimension has a human-readable label — the meaning is **distributed** across all of them together.

**What the bar chart shows:**
- 🟣 **Purple bars (positive values)** — this feature is present / active for this chunk
- 🔴 **Red bars (negative values)** — this feature is suppressed / absent

**Why does the norm matter?**
The *norm* (length) of a vector tells you its magnitude. For **cosine similarity**, only the
*direction* matters — two vectors pointing the same way are considered similar even if one is
twice as long. That's why cosine is the standard for text embeddings.

**Key insight:**
Two chunks that discuss similar topics will produce vectors that point in nearly the same
direction in this {dim}-dimensional space — even if they use different words.
""")

            embed_prog    = st.progress(0)
            embed_explain = st.empty()
            vec_bar_slot  = st.empty()
            dim_display   = st.empty()

            for i, chunk in enumerate(chunks):
                embed_prog.progress((i + 1) / len(chunks),
                                    text=f"Embedding chunk {i+1}/{len(chunks)}…")

                vec = embed(chunk["text"], model, dim)
                all_vectors.append(vec)
                all_chunks.append(chunk)

                pos_dims = int((vec > 0).sum())
                neg_dims = int((vec < 0).sum())
                norm_val = float(np.linalg.norm(vec))

                with embed_explain.container():
                    st.info(
                        f"**Chunk {i+1} → Vector** · "
                        f"The model read *\"{chunk['text'][:70].strip()}...\"* and produced a "
                        f"**{dim}-dimensional vector**. "
                        f"{pos_dims} dimensions are positive (features present), "
                        f"{neg_dims} are negative (features absent). "
                        f"Vector norm = `{norm_val:.4f}`."
                    )

                vec_bar_slot.plotly_chart(
                    vec_bar_fig(vec),
                    use_container_width=True,
                    key=f"vbar_{i}",
                )
                dim_display.caption(
                    f"dim={dim} | positive dims={pos_dims} | negative dims={neg_dims} | "
                    f"min={vec.min():.4f} | max={vec.max():.4f} | "
                    f"mean={vec.mean():.4f} | norm={norm_val:.4f}"
                )
                time.sleep(anim_speed * 0.55)

            embed_prog.progress(1.0, text=f"✅ {len(chunks)} embeddings computed — dim={dim}")
            embed_explain.empty()

            st.success(
                f"✅ **Embedding complete.** Each of the {len(chunks)} chunks is now a "
                f"**{dim}-dimensional vector**. Semantically similar chunks will be close "
                f"to each other in this high-dimensional space."
            )

            st.divider()

            # ── STEP 3: Storing in vector space ───────────────────────
            st.markdown('<div class="step-header">🗄️ Step 3 — Storing Vectors in VecDB</div>',
                        unsafe_allow_html=True)

            st.markdown(f"""
> **What is a vector database?**
>
> A vector database stores embeddings and lets you search them by *similarity* — not by keyword.
> When you query *"what caused the error?"*, it finds chunks whose vectors are **closest in direction**
> to your query's vector, even if none of them contain the exact words you typed.
>
> We're using **{index_type.upper()}** index with **{metric}** distance.
""")

            with st.expander(f"📖 How does the {index_type.upper()} index work?"):
                if index_type == "hnsw":
                    st.markdown("""
**HNSW — Hierarchical Navigable Small World**

Imagine a multi-floor building. Each floor is a graph where nodes are vectors and edges connect similar ones.

- **Top floors** have very few nodes — only the most "landmark" vectors. Searching here is fast.
- **Lower floors** have more nodes. Each floor adds finer detail.
- **Floor 0** has ALL vectors connected to their nearest neighbors.

**Insertion (what you're watching now):**
1. A new vector arrives → assigned a random maximum floor
2. Starting at the top floor, we greedily walk toward the new vector
3. At each floor, we connect the new node to its nearest neighbors
4. This builds a "navigable small world" — any two nodes are reachable in O(log n) hops

**Search:**
Enter at top → descend greedily → beam-search floor 0 → return top-k

**Why HNSW?**
It achieves near-exact recall at a fraction of the cost of brute force.
Qdrant, Weaviate, and Pinecone all use HNSW under the hood.
""")
                else:
                    st.markdown("""
**Flat Index — Brute Force**

Every search compares the query vector against **every single stored vector**.

- ✅ 100% accurate — never misses the true nearest neighbor
- ✅ Simple — no index structure to build
- ❌ Slow at scale — O(n) per query. 1M vectors = 1M dot products per search

Good for: small datasets (<10k vectors), testing, or when you need guaranteed exact results.

For production at scale, switch to HNSW.
""")

            with st.expander("📖 What is PCA and why does the scatter plot look the way it does?"):
                st.markdown(f"""
**The problem:** Your vectors have **{dim} dimensions**. A human screen has 2.
We need to project from {dim}D → 2D to visualize.

**PCA (Principal Component Analysis)** finds the 2 directions in {dim}-dimensional space
along which the data varies the most, and projects everything onto those 2 axes.

**What the scatter plot means:**
- Each **dot = one chunk**
- **Distance between dots ≈ semantic distance** — chunks about similar topics cluster together
- The axes (PC1, PC2) are abstract combinations of the original {dim} dimensions — not directly interpretable
- **Clusters** you see = groups of chunks that discuss the same topic

**Important caveat:** PCA loses a lot of information going from {dim}D to 2D.
Two chunks that look far apart on the plot might actually be close in the real {dim}D space.
The scatter is for intuition, not precision.
""")

            store_prog   = st.progress(0)
            store_explain = st.empty()
            scatter_slot = st.empty()

            for i in range(len(all_chunks)):
                vid = f"chunk_{i}"
                col_obj.insert(vid, all_vectors[i].tolist(), {
                    "chunk_index": i,
                    "text": all_chunks[i]["text"],
                    "start": all_chunks[i]["start"],
                    "end": all_chunks[i]["end"],
                })

                store_prog.progress((i + 1) / len(all_chunks),
                                    text=f"Stored {i+1}/{len(all_chunks)} vectors")

                coords = pca_2d(all_vectors[:i + 1])

                position_note = ""
                if i > 0:
                    # Find closest already-stored chunk
                    sims = [
                        float(np.dot(all_vectors[i], all_vectors[j]) /
                              (np.linalg.norm(all_vectors[i]) * np.linalg.norm(all_vectors[j]) + 1e-9))
                        for j in range(i)
                    ]
                    closest_idx = int(np.argmax(sims))
                    closest_sim = sims[closest_idx]
                    position_note = (
                        f" Its nearest neighbor so far is **Chunk {closest_idx+1}** "
                        f"(cosine similarity = `{closest_sim:.4f}`). "
                        + ("They discuss similar content." if closest_sim > 0.8
                           else "They have moderate semantic overlap." if closest_sim > 0.6
                           else "They cover different topics.")
                    )

                with store_explain.container():
                    st.info(
                        f"**Storing chunk {i+1}** as `chunk_{i}` in the `{index_type.upper()}` index. "
                        f"The ⭐ gold star shows where this vector lands in the 2D projection."
                        + position_note
                    )

                scatter_slot.plotly_chart(
                    scatter_fig(coords, all_chunks[:i + 1], highlight_last=True),
                    use_container_width=True,
                    key=f"scatter_{i}",
                )
                time.sleep(anim_speed * 0.45)

            store_prog.progress(1.0, text=f"✅ All {len(all_chunks)} vectors stored!")
            store_explain.empty()

            # Persist to session state
            st.session_state.chunks  = all_chunks
            st.session_state.vectors = all_vectors
            st.session_state.processed = True

            st.success(
                f"✅ **Storage complete.** {len(all_chunks)} vectors are now indexed in VecDB. "
                f"You can now search this collection by meaning — not just keywords."
            )

            st.markdown("""
---
### 🎓 What just happened — end to end

| Step | Input | Process | Output |
|------|-------|---------|--------|
| 1. Chunking | Raw document | Split by size + overlap | {n} text chunks |
| 2. Embedding | Text chunks | `all-MiniLM-L6-v2` model | {n} vectors of dim={dim} |
| 3. Indexing | Vectors | Build `{idx}` graph | Searchable vector index |

**Now when you search:**
1. Your query text → embedding model → query vector
2. Query vector → `{idx}` index → nearest neighbors in {dim}D space
3. Return the top-k most semantically similar chunks
""".format(n=len(all_chunks), dim=dim, idx=index_type.upper()))

# ══════════════════════════════════════════════════════════════════════════════
# PERSISTENT RESULTS — shown after processing, survives every rerun
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.processed:
    COLORS = [
        "#7c3aed", "#0ea5e9", "#10b981", "#f59e0b",
        "#ef4444", "#ec4899", "#14b8a6", "#8b5cf6",
    ]
    all_chunks  = st.session_state.chunks
    all_vectors = st.session_state.vectors
    dim         = len(all_vectors[0])
    model       = load_model() if use_model else None
    col_obj     = st.session_state.db.get_collection(st.session_state.collection_name)

    st.divider()
    tab_chunks, tab_vecs, tab_space = st.tabs(["📄 Chunks", "🔢 Vectors", "🌐 Vector Space"])

    # ── Tab: Chunks ────────────────────────────────────────────────────────────
    with tab_chunks:
        st.subheader(f"📄 {len(all_chunks)} Chunks")
        for i, chunk in enumerate(all_chunks):
            color = COLORS[i % len(COLORS)]
            with st.expander(
                f"Chunk {i+1}  ·  chars {chunk['start']}–{chunk['end']}  ·  {len(chunk['text'])} chars"
            ):
                st.markdown(
                    f"<div class='chunk-box' style='border-left-color:{color}'>"
                    f"{chunk['text']}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Words: {len(chunk['text'].split())}  |  "
                    f"Start: {chunk['start']}  |  End: {chunk['end']}"
                )

    # ── Tab: Vectors ───────────────────────────────────────────────────────────
    with tab_vecs:
        st.subheader(f"🔢 {len(all_vectors)} Vectors  ·  dim = {dim}")
        for i, (chunk, vec) in enumerate(zip(all_chunks, all_vectors)):
            label = chunk['text'][:70].strip()
            with st.expander(f"Vector {i+1}  ·  _{label}..._"):
                l, r = st.columns([1, 2])
                with l:
                    st.caption("First 20 dimensions")
                    for d, val in enumerate(vec[:20]):
                        bar_w = int(abs(val) * 100)
                        color = "#7c3aed" if val >= 0 else "#ef4444"
                        st.markdown(
                            f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0'>"
                            f"<span style='color:#6b7280;font-size:11px;width:50px'>dim[{d:02d}]</span>"
                            f"<div style='flex:1;background:#1e2235;height:6px;border-radius:3px'>"
                            f"<div style='width:{bar_w}%;background:{color};height:100%;border-radius:3px'></div>"
                            f"</div>"
                            f"<span style='color:#e2e8f0;font-size:11px;width:80px'>{val:.6f}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                with r:
                    st.caption("All dimensions (bar chart)")
                    st.plotly_chart(
                        vec_bar_fig(vec, n=min(128, dim), height=150),
                        use_container_width=True,
                        key=f"vbar_tab_{i}",
                    )
                    st.caption(
                        f"norm={np.linalg.norm(vec):.4f}  |  "
                        f"min={vec.min():.4f}  |  max={vec.max():.4f}  |  "
                        f"mean={vec.mean():.4f}"
                    )

    # ── Tab: Vector Space + Search ─────────────────────────────────────────────
    with tab_space:
        st.subheader("🌐 Vector Space — PCA 2D Projection")

        if len(all_vectors) >= 2:
            coords = pca_2d(all_vectors)

            try:
                from sklearn.decomposition import PCA as _PCA
                pca_obj = _PCA(n_components=2).fit(np.array(all_vectors, dtype=np.float32))
                ev = pca_obj.explained_variance_ratio_
                st.caption(
                    f"PCA variance explained → PC1: {ev[0]:.1%}  |  PC2: {ev[1]:.1%}  |  "
                    f"Combined: {ev.sum():.1%}"
                )
            except ImportError:
                st.caption("(sklearn not available — using manual SVD projection)")

            st.plotly_chart(scatter_fig(coords, all_chunks, height=500), use_container_width=True)

            st.divider()
            st.subheader("🔍 Search the Vector Space")

            # ── Metric description ─────────────────────────────────────
            METRIC_INFO = {
                "cosine": {
                    "icon": "📐",
                    "title": "Cosine Similarity",
                    "formula": "score = (q · v) / (‖q‖ × ‖v‖)",
                    "range": "−1.0 to +1.0  (higher = more similar)",
                    "how": (
                        "Measures the **angle** between your query vector and each stored vector. "
                        "Two vectors pointing in the same direction score **1.0** (identical meaning), "
                        "perpendicular vectors score **0.0** (unrelated), "
                        "opposite vectors score **−1.0**."
                    ),
                    "why": (
                        "Ignores vector magnitude — only direction matters. "
                        "This makes it ideal for text: a short sentence and a long paragraph "
                        "about the same topic will still score high, because their *direction* "
                        "in embedding space is similar even if their lengths differ."
                    ),
                    "example": "score(1.0) = identical  |  score(0.8+) = very similar  |  score(<0.5) = unrelated",
                },
                "euclidean": {
                    "icon": "📏",
                    "title": "Euclidean Distance (L2)",
                    "formula": "score = −‖q − v‖₂  (negative so higher = closer)",
                    "range": "−∞ to 0  (closer to 0 = more similar)",
                    "how": (
                        "Measures the **straight-line distance** between two points in 384-dimensional space. "
                        "A score of **0** means the vectors are identical. "
                        "We negate it so a higher score still means more similar — consistent with cosine."
                    ),
                    "why": (
                        "Sensitive to vector magnitude, unlike cosine. "
                        "If one chunk's embedding has a much larger norm than another, "
                        "it will appear far away even if the topic is the same. "
                        "Best used when vectors are normalized or when absolute scale matters."
                    ),
                    "example": "score(0.0) = identical  |  score(−1 to −3) = similar  |  score(< −5) = different",
                },
                "dot": {
                    "icon": "✖️",
                    "title": "Dot Product",
                    "formula": "score = q · v = Σ(qᵢ × vᵢ)",
                    "range": "−∞ to +∞  (higher = more similar)",
                    "how": (
                        "Multiplies each dimension of the query and chunk vectors together and sums the result. "
                        "Unlike cosine it is **not normalized** — both direction AND magnitude affect the score. "
                        "A large, similar vector scores higher than a small, similar vector."
                    ),
                    "why": (
                        "Fastest to compute (no normalization step). "
                        "Used in recommendation systems where the *magnitude* of an embedding "
                        "encodes importance (e.g. a popular item has a larger embedding norm). "
                        "For plain text embeddings, cosine is usually better because magnitude is arbitrary."
                    ),
                    "example": "Higher scores = more similar AND longer vector norm",
                },
            }

            info = METRIC_INFO.get(metric, METRIC_INFO["cosine"])
            st.markdown(
                f"<div style='background:#111827;border:1px solid #1e2235;border-left:4px solid #7c3aed;"
                f"border-radius:8px;padding:16px 20px;margin-bottom:16px'>"
                f"<div style='color:#a78bfa;font-weight:700;font-size:15px;margin-bottom:8px'>"
                f"{info['icon']} Active Metric: {info['title']}</div>"
                f"<div style='font-family:monospace;background:#0b0d14;padding:8px 12px;border-radius:4px;"
                f"color:#34d399;margin-bottom:10px;font-size:13px'>{info['formula']}</div>"
                f"<div style='color:#94a3b8;font-size:13px;margin-bottom:6px'>"
                f"<b style='color:#e2e8f0'>Range:</b> {info['range']}</div>"
                f"<div style='color:#94a3b8;font-size:13px;margin-bottom:6px'>"
                f"<b style='color:#e2e8f0'>How it works:</b> {info['how']}</div>"
                f"<div style='color:#94a3b8;font-size:13px;margin-bottom:6px'>"
                f"<b style='color:#e2e8f0'>Why use it:</b> {info['why']}</div>"
                f"<div style='color:#64748b;font-size:12px;font-style:italic'>{info['example']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            query = st.text_input("Enter a query:", placeholder="What is this document about?")
            k_val = st.slider("Top K results", 1, min(10, len(all_chunks)), 3)

            if query:
                q_vec_arr = embed(query, model, dim)
                q_vec = q_vec_arr.tolist()
                results = col_obj.search(q_vec, k=k_val, include_vector=True)

                # ── Scatter with highlights ────────────────────────────
                match_indices = [r["metadata"]["chunk_index"] for r in results]
                fig2 = scatter_fig(coords, all_chunks, height=420)
                fig2.add_trace(go.Scatter(
                    x=coords[match_indices, 0],
                    y=coords[match_indices, 1],
                    mode="markers",
                    marker=dict(size=24, color="rgba(251,191,36,0.25)",
                                line=dict(color="#fbbf24", width=3), symbol="circle"),
                    hoverinfo="skip", name="matches",
                ))
                # Add query point if PCA is available
                try:
                    from sklearn.decomposition import PCA as _PCA
                    pca_q = _PCA(n_components=2).fit(np.array(all_vectors, dtype=np.float32))
                    q_coord = pca_q.transform(q_vec_arr.reshape(1, -1))
                    fig2.add_trace(go.Scatter(
                        x=[q_coord[0, 0]], y=[q_coord[0, 1]],
                        mode="markers+text",
                        marker=dict(size=18, color="#f87171", symbol="diamond",
                                    line=dict(color="white", width=2)),
                        text=["Query"], textposition="top center",
                        textfont=dict(color="#f87171", size=12),
                        hovertext=f"Query: {query[:60]}", hoverinfo="text",
                        name="query",
                    ))
                except Exception:
                    pass
                st.plotly_chart(fig2, use_container_width=True, key="search_scatter")
                st.caption("🔴 Red diamond = your query vector  |  🟡 Gold circles = top-k matches")

                # ── Results cards ──────────────────────────────────────
                st.markdown(f"**Top {len(results)} results using `{metric}` metric:**")

                for rank, r in enumerate(results, 1):
                    chunk_idx = r["metadata"]["chunk_index"]
                    score = r["score"]
                    chunk_vec = np.array(r.get("vector") or all_vectors[chunk_idx], dtype=np.float32)

                    # Score interpretation
                    if metric == "cosine":
                        if score >= 0.9:   interp, color = "Extremely similar", "#10b981"
                        elif score >= 0.75: interp, color = "Highly similar",    "#34d399"
                        elif score >= 0.6:  interp, color = "Moderately similar","#f59e0b"
                        else:               interp, color = "Loosely related",   "#ef4444"
                    elif metric == "euclidean":
                        if score >= -1:    interp, color = "Very close",         "#10b981"
                        elif score >= -3:  interp, color = "Nearby",             "#34d399"
                        elif score >= -6:  interp, color = "Moderate distance",  "#f59e0b"
                        else:              interp, color = "Far apart",           "#ef4444"
                    else:  # dot
                        if score >= 50:    interp, color = "Strong match",       "#10b981"
                        elif score >= 20:  interp, color = "Good match",         "#34d399"
                        elif score >= 5:   interp, color = "Weak match",         "#f59e0b"
                        else:              interp, color = "Poor match",          "#ef4444"

                    with st.expander(
                        f"#{rank}  ·  Score: {score:.4f}  ·  {interp}  ·  Chunk {chunk_idx + 1}"
                    ):
                        # Score explanation
                        st.markdown(
                            f"<div style='background:#0b0d14;border-left:3px solid {color};"
                            f"border-radius:4px;padding:10px 14px;margin-bottom:12px'>"
                            f"<span style='color:{color};font-weight:600'>{interp}</span>"
                            f"<span style='color:#6b7280;font-size:12px'> — "
                            f"{info['title']} score of <code>{score:.4f}</code>. "
                            + (
                                f"The angle between query and this chunk is "
                                f"<code>{np.degrees(np.arccos(min(1.0, max(-1.0, score)))):.1f}°</code>."
                                if metric == "cosine" else
                                f"Straight-line distance = <code>{abs(score):.4f}</code>."
                                if metric == "euclidean" else
                                f"Raw dot product = <code>{score:.4f}</code>."
                            )
                            + "</span></div>",
                            unsafe_allow_html=True,
                        )

                        # Chunk text
                        st.code(r["metadata"]["text"], language=None)

                        # Vector comparison
                        st.markdown("**Query vs Chunk — first 32 dimensions:**")
                        comp_fig = go.Figure()
                        dims = list(range(32))
                        comp_fig.add_trace(go.Bar(
                            x=dims, y=q_vec_arr[:32].tolist(),
                            name="Query", marker_color="#f87171", opacity=0.8,
                        ))
                        comp_fig.add_trace(go.Bar(
                            x=dims, y=chunk_vec[:32].tolist(),
                            name=f"Chunk {chunk_idx+1}", marker_color="#7c3aed", opacity=0.8,
                        ))
                        comp_fig.update_layout(
                            height=160, barmode="overlay",
                            margin=dict(l=0, r=0, t=4, b=0),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(showticklabels=False, showgrid=False),
                            yaxis=dict(showgrid=False, color="#4b5563"),
                            legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
                            font=dict(color="#e2e8f0"),
                        )
                        st.plotly_chart(comp_fig, use_container_width=True, key=f"comp_{rank}")
                        st.caption(
                            f"Overlapping bars = dimensions where query and chunk agree. "
                            f"Vector norm — Query: {np.linalg.norm(q_vec_arr):.4f}  |  "
                            f"Chunk: {np.linalg.norm(chunk_vec):.4f}"
                        )
        else:
            st.info("Process a document with at least 2 chunks to see the vector space.")

# ─── Empty state ───────────────────────────────────────────────────────────────
if not uploaded_file and not st.session_state.processed:
    st.markdown("""
    <div style='text-align:center;padding:80px 20px;color:#4b5563'>
        <div style='font-size:56px;margin-bottom:16px'>🔮</div>
        <div style='font-size:20px;color:#94a3b8;margin-bottom:8px'>Upload a document to get started</div>
        <div style='font-size:14px'>Supports TXT and PDF · Watch chunking, embedding & storage happen live</div>
    </div>
    """, unsafe_allow_html=True)
