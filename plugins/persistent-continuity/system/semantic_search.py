#!/usr/bin/env python3
"""
semantic_search.py — Retrieve semantic memory facts by meaning, not just keywords.

Operates in two modes:
  Mode 1 (default): TF-IDF bag-of-words retrieval — zero dependencies, pure stdlib
  Mode 2 (optional): Dense vector embeddings via sentence-transformers (pip install)

Commands:
    [query]              Search facts by meaning: python semantic_search.py "auth patterns"
    --list               List all facts with IDs and concepts
    --add                Interactive: add a new fact
    --show FACT-NNN      Show full fact file
    --tag [tag ...]      Filter by tags before scoring
    --confidence [lvl]   Filter by confidence (high/medium/low)
    --reindex            Rebuild TF-IDF index (auto on first search)
    --embed              Build dense embedding index (requires sentence-transformers)
    --top N              Return top N results (default: 5)

Usage:
    python semantic_search.py "how does session pooling work"
    python semantic_search.py --tag python --tag async "concurrency patterns"
    python semantic_search.py --confidence high "authentication"
    python semantic_search.py --add
    python semantic_search.py --list
"""

import os
import sys
import re
import math
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import data_root
# Data root is decoupled from this code (see _paths.py): per-project .continuity/
# by default, or $CONTINUITY_HOME. Plugin code itself is shared/read-only.
ROOT = data_root()
SEMANTIC_DIR = ROOT / "semantic"
FACTS_DIR = SEMANTIC_DIR / "facts"
EMBEDDINGS_DIR = SEMANTIC_DIR / "embeddings"
INDEX_FILE = SEMANTIC_DIR / "SEMANTIC_INDEX.md"
TFIDF_CACHE = EMBEDDINGS_DIR / "tfidf_index.json"


# ─── FACT PARSING ──────────────────────────────────────────────────────────

def parse_fact(filepath: Path) -> dict:
    """Parse a fact file into a dict."""
    content = filepath.read_text(encoding="utf-8")
    fact = {"_path": str(filepath.relative_to(ROOT)), "_raw": content}

    # Parse frontmatter
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            fm = content[3:end]
            for line in fm.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    k, v = k.strip(), v.strip()
                    if v.startswith("[") and v.endswith("]"):
                        v = [x.strip().strip('"\'') for x in v[1:-1].split(",") if x.strip()]
                    elif v.startswith('"') and v.endswith('"'):
                        v = v[1:-1]
                    fact[k] = v
            body = content[end + 4:].strip()
        else:
            body = content
    else:
        body = content

    fact["_body"] = body

    # Get statement (first non-heading paragraph in body)
    lines = body.splitlines()
    statement_lines = []
    in_statement = False
    for line in lines:
        if line.startswith("## Statement"):
            in_statement = True
            continue
        if in_statement:
            if line.startswith("##"):
                break
            if line.strip():
                statement_lines.append(line.strip())
    fact["_statement"] = " ".join(statement_lines[:3])

    return fact


def load_all_facts() -> list[dict]:
    """Load all fact files from semantic/facts/."""
    facts = []
    for f in sorted(FACTS_DIR.glob("FACT-*.md")):
        try:
            facts.append(parse_fact(f))
        except Exception as e:
            print(f"  WARN: Could not parse {f.name}: {e}")
    return facts


# ─── TF-IDF ENGINE ─────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s_-]', ' ', text)
    tokens = text.split()
    # Remove very common stop words
    stops = {
        'the', 'a', 'an', 'is', 'it', 'in', 'on', 'at', 'to', 'for',
        'of', 'and', 'or', 'but', 'not', 'with', 'this', 'that', 'are',
        'was', 'be', 'by', 'from', 'as', 'into', 'about', 'used', 'can',
        'will', 'has', 'have', 'had', 'its', 'which', 'when', 'how', 'what',
    }
    return [t for t in tokens if t not in stops and len(t) > 1]


def build_tfidf_index(facts: list[dict]) -> dict:
    """Build a TF-IDF index from facts. Returns {term: {fact_id: tfidf_score}}."""
    # Build term frequency per document
    doc_terms = {}
    for fact in facts:
        fid = fact.get("id", fact["_path"])
        text = " ".join([
            fact.get("concept", ""),
            fact.get("_statement", ""),
            " ".join(fact.get("tags", [])),
            fact.get("_body", "")[:500],
        ])
        tokens = tokenize(text)
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        # Normalize by document length
        total = max(sum(tf.values()), 1)
        doc_terms[fid] = {t: c / total for t, c in tf.items()}

    # Compute IDF
    N = max(len(doc_terms), 1)
    df = defaultdict(int)
    for terms in doc_terms.values():
        for t in terms:
            df[t] += 1
    idf = {t: math.log(N / (df[t] + 1)) + 1 for t in df}

    # Build inverted index: term -> {fact_id: tfidf}
    inv_index = defaultdict(dict)
    for fid, terms in doc_terms.items():
        for t, tf_score in terms.items():
            inv_index[t][fid] = tf_score * idf.get(t, 1.0)

    return dict(inv_index)


def tfidf_search(query: str, facts: list[dict], top_n: int = 5) -> list[tuple[dict, float]]:
    """Return top_n facts ranked by TF-IDF similarity to query."""
    index = build_tfidf_index(facts)
    query_tokens = tokenize(query)

    scores = defaultdict(float)
    for token in query_tokens:
        if token in index:
            for fid, score in index[token].items():
                scores[fid] += score

    # Map back to fact dicts
    fact_by_id = {f.get("id", f["_path"]): f for f in facts}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for fid, score in ranked[:top_n]:
        if fid in fact_by_id:
            results.append((fact_by_id[fid], score))
    return results


# ─── DENSE EMBEDDING ENGINE (OPTIONAL) ────────────────────────────────────

def try_dense_search(query: str, facts: list[dict], top_n: int = 5):
    """Attempt dense embedding search. Falls back to TF-IDF if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        return None  # Signal fallback

    model_name = "all-MiniLM-L6-v2"
    embed_file = EMBEDDINGS_DIR / "embeddings.npy"
    meta_file = EMBEDDINGS_DIR / "index.json"

    if not embed_file.exists() or not meta_file.exists():
        print("  Embedding index not found. Run --embed to build it.")
        return None

    import numpy as np
    embeddings = np.load(str(embed_file))
    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)

    model = SentenceTransformer(model_name)
    q_vec = model.encode([query])[0]
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    normed = embeddings / norms
    scores = normed @ q_norm

    top_idx = np.argsort(scores)[::-1][:top_n]
    fact_by_id = {f.get("id"): f for f in facts}
    results = []
    for i in top_idx:
        fid = meta[i]["id"]
        if fid in fact_by_id:
            results.append((fact_by_id[fid], float(scores[i])))
    return results


def build_embeddings(facts: list[dict]):
    """Build and save dense embeddings for all facts."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        print("  ERROR: sentence-transformers not installed.")
        print("  Install with: pip install sentence-transformers")
        sys.exit(1)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = []
    meta = []
    for fact in facts:
        text = f"{fact.get('concept', '')}. {fact.get('_statement', '')}"
        texts.append(text)
        meta.append({"id": fact.get("id"), "concept": fact.get("concept")})

    print(f"  Encoding {len(texts)} facts...")
    embeddings = model.encode(texts, show_progress_bar=True)

    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    import numpy as np
    np.save(str(EMBEDDINGS_DIR / "embeddings.npy"), embeddings)
    with open(EMBEDDINGS_DIR / "index.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Embedding index saved to {EMBEDDINGS_DIR.relative_to(ROOT)}")


# ─── DISPLAY ───────────────────────────────────────────────────────────────

def print_results(results: list[tuple[dict, float]], mode: str = "tfidf"):
    if not results:
        print("  No matching facts found.")
        return

    print(f"\n  Top results (mode: {mode}):\n")
    for rank, (fact, score) in enumerate(results, 1):
        fid = fact.get("id", "?")
        concept = fact.get("concept", "?")
        confidence = fact.get("confidence", "?")
        tags = fact.get("tags", [])
        tag_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
        statement = fact.get("_statement", "")[:120]
        print(f"  [{rank}] {fid} (score: {score:.3f})")
        print(f"      Concept:    {concept}")
        print(f"      Confidence: {confidence}  |  Tags: {tag_str}")
        print(f"      Statement:  {statement}")
        print(f"      File:       {fact.get('_path', '?')}")
        print()


def cmd_list(facts: list[dict]):
    if not facts:
        print("  No facts found in semantic/facts/")
        return
    print(f"\n  All facts ({len(facts)} total):\n")
    print(f"  {'ID':<15} {'Confidence':<12} {'Tags':<30} Concept")
    print(f"  {'-'*15} {'-'*12} {'-'*30} {'-'*30}")
    for fact in facts:
        fid = fact.get("id", "?")
        conf = fact.get("confidence", "?")
        tags = fact.get("tags", [])
        tag_str = (", ".join(tags) if isinstance(tags, list) else str(tags))[:28]
        concept = (fact.get("concept", "?") or "?")[:40]
        print(f"  {fid:<15} {conf:<12} {tag_str:<30} {concept}")


def cmd_show(fact_id: str, facts: list[dict]):
    for fact in facts:
        if fact.get("id") == fact_id:
            print(f"\n  {fact['_path']}\n")
            print(fact["_raw"])
            return
    print(f"  Fact not found: {fact_id}")


def cmd_add():
    """Interactive fact creation."""
    print("\n  Adding new fact to semantic memory\n")
    fid_num = len(list(FACTS_DIR.glob("FACT-*.md"))) + 1
    fid = f"FACT-{fid_num:03d}"
    print(f"  Fact ID: {fid}")

    concept = input("  Concept (one-line name): ").strip()
    statement = input("  Statement (the fact itself, 1-3 sentences): ").strip()
    tags_raw = input("  Tags (comma-separated): ").strip()
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    confidence = input("  Confidence [high/medium/low] (default: medium): ").strip() or "medium"
    source = input("  Source (file path, session, or external): ").strip()
    date = __import__("datetime").date.today().isoformat()

    tag_str = "[" + ", ".join(tags) + "]"
    content = f"""---
id: {fid}
concept: "{concept}"
confidence: {confidence}
created: {date}
updated: {date}
tags: {tag_str}
source: "{source}"
related_facts: []
embedding: null
---

# {fid}: {concept}

## Statement

{statement}

## Context

[When is this fact relevant? Any caveats or conditions?]

## Evidence / Source Detail

Source: {source}

## Related Facts

*(none yet)*

## Applications

*(document as the fact is applied)*
"""
    out_path = FACTS_DIR / f"{fid}_{re.sub(r'[^a-z0-9]+', '-', concept.lower())[:40]}.md"
    out_path.write_text(content, encoding="utf-8")
    print(f"\n  Fact created: {out_path.relative_to(ROOT)}")
    print("  Remember to update semantic/SEMANTIC_INDEX.md")


# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return

    # Parse flags
    top_n = 5
    tag_filters = []
    confidence_filter = None
    query_parts = []
    do_list = do_add = do_embed = do_reindex = False
    show_id = None

    i = 0
    while i < len(args):
        if args[i] == "--list":
            do_list = True
        elif args[i] == "--add":
            do_add = True
        elif args[i] == "--embed":
            do_embed = True
        elif args[i] == "--reindex":
            do_reindex = True
        elif args[i] == "--top" and i + 1 < len(args):
            top_n = int(args[i + 1]); i += 1
        elif args[i] == "--tag" and i + 1 < len(args):
            tag_filters.append(args[i + 1].lower()); i += 1
        elif args[i] == "--confidence" and i + 1 < len(args):
            confidence_filter = args[i + 1].lower(); i += 1
        elif args[i] == "--show" and i + 1 < len(args):
            show_id = args[i + 1]; i += 1
        else:
            query_parts.append(args[i])
        i += 1

    facts = load_all_facts()

    # Apply filters
    if tag_filters:
        facts = [
            f for f in facts
            if all(
                t in (f.get("tags") or [])
                for t in tag_filters
            )
        ]
    if confidence_filter:
        facts = [f for f in facts if f.get("confidence") == confidence_filter]

    if do_add:
        cmd_add()
    elif do_list:
        cmd_list(facts)
    elif show_id:
        cmd_show(show_id, facts)
    elif do_embed:
        build_embeddings(facts)
    elif query_parts:
        query = " ".join(query_parts)
        print(f"\n  Semantic search: \"{query}\"")
        if tag_filters:
            print(f"  Tag filters: {tag_filters}")
        if confidence_filter:
            print(f"  Confidence: {confidence_filter}")
        print(f"  Facts in scope: {len(facts)}")

        # Try dense first, fall back to TF-IDF
        results = try_dense_search(query, facts, top_n)
        mode = "dense-embedding"
        if results is None:
            results = tfidf_search(query, facts, top_n)
            mode = "tfidf"

        print_results(results, mode)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
