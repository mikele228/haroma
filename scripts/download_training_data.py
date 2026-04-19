#!/usr/bin/env python3
"""
Download public training data and import it into HaromaX6's cognitive modules.

Sources (all freely available):
  1. ConceptNet 5  — common-sense knowledge triples (CC-BY-SA 4.0)
     https://github.com/commonsense/conceptnet5
  2. NRC Emotion Lexicon (EmoLex) — word-emotion associations
     https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
  3. English word frequency (Hermit Dave / OpenSubtitles) — top 10k words
     https://github.com/hermitdave/FrequencyWords
  4. WordNet via NLTK — semantic relationships
  5. DailyDialog — conversational dataset (CC-BY-NC-SA 4.0)
     http://yanran.li/dailydialog

After download, the script imports data into:
  - KnowledgeGraph entities + relations  (ConceptNet + WordNet)
  - NLU emotion lexicon                  (NRC EmoLex)
  - LanguageComposer word vocabulary     (frequency list)
  - Training samples for language model  (DailyDialog)
"""

import os
import sys
import json
import gzip
import time
import hashlib
import urllib.request
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DOWNLOAD_DIR = os.path.join(DATA_DIR, "downloads")
TRAINING_DIR = os.path.join(DATA_DIR, "training")
COGNITIVE_DIR = os.path.join(DATA_DIR, "cognitive")

# Set True from main() when --quick (smaller downloads for dev machines)
_DOWNLOAD_QUICK = False

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(COGNITIVE_DIR, exist_ok=True)


def _cap_conceptnet() -> int:
    return 12_000 if _DOWNLOAD_QUICK else 200_000


def _cap_wordnet() -> int:
    return 2_000 if _DOWNLOAD_QUICK else 50_000


def _cap_dailydialog() -> int:
    return 4_000 if _DOWNLOAD_QUICK else 100_000


def _cap_vocab_dialog_rows() -> int:
    return 3_000 if _DOWNLOAD_QUICK else 50_000


def _download(url: str, dest: str, label: str) -> str:
    if os.path.exists(dest):
        print(f"  [{label}] Already downloaded: {os.path.basename(dest)}")
        return dest
    print(f"  [{label}] Downloading {url[:80]}...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  [{label}] Saved ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  [{label}] Download failed: {e}")
        return ""
    return dest


# =====================================================================
# 1. ConceptNet — common-sense knowledge
# =====================================================================

CONCEPTNET_URL = (
    "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
)

# Predicates we care about (maps ConceptNet relation URIs to short names)
CN_PREDICATES = {
    "/r/IsA": "is_a",
    "/r/PartOf": "part_of",
    "/r/HasA": "has_a",
    "/r/UsedFor": "used_for",
    "/r/CapableOf": "capable_of",
    "/r/Causes": "causes",
    "/r/HasProperty": "has_property",
    "/r/MotivatedByGoal": "motivated_by",
    "/r/DefinedAs": "defined_as",
    "/r/MadeOf": "made_of",
    "/r/ReceivesAction": "receives_action",
    "/r/AtLocation": "at_location",
    "/r/HasSubevent": "has_subevent",
    "/r/HasPrerequisite": "has_prerequisite",
    "/r/Desires": "desires",
    "/r/CausesDesire": "causes_desire",
    "/r/SymbolOf": "symbol_of",
    "/r/RelatedTo": "related_to",
    "/r/Antonym": "antonym",
    "/r/Synonym": "synonym",
}


def download_conceptnet():
    print("\n--- ConceptNet 5.7 ---")
    dest = os.path.join(DOWNLOAD_DIR, "conceptnet-assertions-5.7.0.csv.gz")
    path = _download(CONCEPTNET_URL, dest, "ConceptNet")
    if not path:
        return []

    print("  [ConceptNet] Parsing English triples (this may take a few minutes)...")
    triples = []
    count = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue
                relation = parts[1]
                subj = parts[2]
                obj = parts[3]

                if not subj.startswith("/c/en/") or not obj.startswith("/c/en/"):
                    continue
                if relation not in CN_PREDICATES:
                    continue

                subj_name = subj.split("/")[3].replace("_", " ")
                obj_name = obj.split("/")[3].replace("_", " ")
                pred = CN_PREDICATES[relation]

                try:
                    info = json.loads(parts[4])
                    weight = info.get("weight", 1.0)
                except (json.JSONDecodeError, IndexError):
                    weight = 1.0

                if weight < 1.0:
                    continue

                triples.append(
                    {
                        "subject": subj_name,
                        "predicate": pred,
                        "object": obj_name,
                        "confidence": min(1.0, weight / 5.0),
                        "source": "conceptnet",
                    }
                )
                count += 1
                if count >= _cap_conceptnet():
                    break
    except Exception as e:
        print(f"  [ConceptNet] Parse error: {e}")

    out = os.path.join(TRAINING_DIR, "conceptnet_triples.json")
    if triples:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(triples, f, ensure_ascii=False)
        print(f"  [ConceptNet] Extracted {len(triples):,} English triples -> {out}")
    else:
        print("  [ConceptNet] WARNING: no triples extracted; keeping existing file")
    return triples


# =====================================================================
# 2. NRC Emotion Lexicon
# =====================================================================

EMOLEX_URL = (
    "https://raw.githubusercontent.com/dinbav/LeXmo/master/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
)


def download_emolex():
    print("\n--- NRC Emotion Lexicon ---")
    dest = os.path.join(DOWNLOAD_DIR, "NRC-Emotion-Lexicon.txt")
    path = _download(EMOLEX_URL, dest, "EmoLex")
    if not path:
        return {}

    print("  [EmoLex] Parsing word-emotion associations...")
    lexicon = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                word, emotion, flag = parts[0], parts[1], parts[2]
                if flag != "1":
                    continue
                lexicon.setdefault(word, []).append(emotion)
    except Exception as e:
        print(f"  [EmoLex] Parse error: {e}")

    out = os.path.join(TRAINING_DIR, "emolex.json")
    if lexicon:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(lexicon, f, ensure_ascii=False)
        print(f"  [EmoLex] {len(lexicon):,} words with emotion tags -> {out}")
    else:
        print("  [EmoLex] WARNING: no data extracted; keeping existing file")
    return lexicon


# =====================================================================
# 3. English word frequency list
# =====================================================================

FREQ_URL = (
    "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_50k.txt"
)


def download_word_freq():
    print("\n--- English Word Frequency ---")
    dest = os.path.join(DOWNLOAD_DIR, "en_50k.txt")
    path = _download(FREQ_URL, dest, "FreqWords")
    if not path:
        return []

    print("  [FreqWords] Parsing top words...")
    words = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0].isalpha() and len(parts[0]) > 1:
                    words.append(parts[0].lower())
                if len(words) >= 10000:
                    break
    except Exception as e:
        print(f"  [FreqWords] Parse error: {e}")

    out = os.path.join(TRAINING_DIR, "english_words_10k.json")
    if words:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(words, f, ensure_ascii=False)
        print(f"  [FreqWords] {len(words):,} words -> {out}")
    else:
        print("  [FreqWords] WARNING: no words extracted; keeping existing file")
    return words


# =====================================================================
# 4. WordNet via NLTK (semantic relations)
# =====================================================================


def download_wordnet():
    print("\n--- WordNet (via NLTK) ---")
    try:
        import nltk
    except ImportError:
        print("  [WordNet] Installing NLTK...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk", "--quiet"])
        import nltk

    nltk_data = os.path.join(DOWNLOAD_DIR, "nltk_data")
    nltk.data.path.insert(0, nltk_data)
    os.makedirs(nltk_data, exist_ok=True)

    marker = os.path.join(TRAINING_DIR, "wordnet_triples.json")
    if os.path.exists(marker):
        print("  [WordNet] Already extracted.")
        with open(marker, "r") as f:
            return json.load(f)

    print("  [WordNet] Downloading WordNet data...")
    nltk.download("wordnet", download_dir=nltk_data, quiet=True)
    nltk.download("omw-1.4", download_dir=nltk_data, quiet=True)

    from nltk.corpus import wordnet as wn

    print("  [WordNet] Extracting hypernym/hyponym/meronym relations...")
    triples = []
    seen = set()
    count = 0
    for synset in wn.all_synsets():
        if count >= _cap_wordnet():
            break
        name = synset.lemma_names()[0].replace("_", " ")

        for hyper in synset.hypernyms():
            h_name = hyper.lemma_names()[0].replace("_", " ")
            key = (name, "is_a", h_name)
            if key not in seen:
                seen.add(key)
                triples.append(
                    {
                        "subject": name,
                        "predicate": "is_a",
                        "object": h_name,
                        "confidence": 0.9,
                        "source": "wordnet",
                    }
                )
                count += 1

        for part in synset.part_meronyms():
            p_name = part.lemma_names()[0].replace("_", " ")
            key = (p_name, "part_of", name)
            if key not in seen:
                seen.add(key)
                triples.append(
                    {
                        "subject": p_name,
                        "predicate": "part_of",
                        "object": name,
                        "confidence": 0.85,
                        "source": "wordnet",
                    }
                )
                count += 1

    with open(marker, "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False)
    print(f"  [WordNet] {len(triples):,} triples -> {marker}")
    return triples


# =====================================================================
# 5. DailyDialog — conversational training samples
# =====================================================================

DAILYDIALOG_URL = "https://aclanthology.org/attachments/I17-1099.Datasets.zip"


def download_dailydialog():
    print("\n--- DailyDialog Conversations ---")
    dest = os.path.join(DOWNLOAD_DIR, "dailydialog.zip")
    marker = os.path.join(TRAINING_DIR, "dailydialog_samples.json")
    if os.path.exists(marker):
        sz = os.path.getsize(marker)
        if sz > 100:
            print("  [DailyDialog] Already extracted.")
            with open(marker, "r") as f:
                return json.load(f)

    # Remove stale/corrupt downloads
    if os.path.exists(dest) and os.path.getsize(dest) < 5000:
        os.remove(dest)

    path = _download(DAILYDIALOG_URL, dest, "DailyDialog")
    if not path:
        return _fallback_dailydialog()

    # Validate it's actually a zip
    if os.path.getsize(path) < 5000:
        print(
            "  [DailyDialog] Downloaded file too small (likely redirect page). Trying fallback..."
        )
        os.remove(path)
        return _fallback_dailydialog()

    print("  [DailyDialog] Extracting conversations...")
    samples = []
    try:
        with zipfile.ZipFile(path, "r") as zf:
            all_names = zf.namelist()
            dialog_files = [
                n for n in all_names if "dialogues_text" in n.lower() and n.endswith(".txt")
            ]
            if not dialog_files:
                dialog_files = [
                    n for n in all_names if "dialog" in n.lower() and n.endswith(".txt")
                ]

            emo_map = {
                0: "neutral",
                1: "anger",
                2: "disgust",
                3: "fear",
                4: "joy",
                5: "sadness",
                6: "surprise",
            }

            # Also check for nested zips (some distributions nest another zip inside)
            inner_zips = [n for n in all_names if n.endswith(".zip")]
            if inner_zips and not dialog_files:
                extract_dir = os.path.join(DOWNLOAD_DIR, "dailydialog_extracted")
                zf.extractall(extract_dir)
                for iz in inner_zips:
                    inner_path = os.path.join(extract_dir, iz)
                    if os.path.exists(inner_path):
                        with zipfile.ZipFile(inner_path, "r") as izf:
                            izf.extractall(extract_dir)
                            dialog_files = [
                                os.path.join(extract_dir, n)
                                for n in izf.namelist()
                                if "dialogues_text" in n.lower() and n.endswith(".txt")
                            ]
                for df in dialog_files:
                    samples.extend(_parse_dialog_file(df, None, emo_map))
                    if len(samples) >= _cap_dailydialog():
                        break
            else:
                for df in dialog_files:
                    ef = df.replace("dialogues_text", "dialogues_emotion")
                    with zf.open(df) as f_d:
                        dialogs = f_d.read().decode("utf-8", errors="ignore").strip().splitlines()
                    emotions_lines = []
                    if ef in all_names:
                        with zf.open(ef) as f_e:
                            emotions_lines = (
                                f_e.read().decode("utf-8", errors="ignore").strip().splitlines()
                            )
                    for i, dialog in enumerate(dialogs):
                        turns = [t.strip() for t in dialog.split("__eou__") if t.strip()]
                        emos = []
                        if i < len(emotions_lines):
                            emos = [
                                int(x.strip())
                                for x in emotions_lines[i].split()
                                if x.strip().isdigit()
                            ]
                        for j, turn in enumerate(turns):
                            emo_label = (
                                emo_map.get(emos[j], "neutral") if j < len(emos) else "neutral"
                            )
                            context = " | ".join(turns[max(0, j - 2) : j]) if j > 0 else ""
                            samples.append(
                                {
                                    "text": turn,
                                    "emotion": emo_label,
                                    "context": context,
                                    "turn_index": j,
                                }
                            )
                        if len(samples) >= _cap_dailydialog():
                            break
                    if len(samples) >= _cap_dailydialog():
                        break
    except (zipfile.BadZipFile, Exception) as e:
        print(f"  [DailyDialog] Parse error: {e}")
        if not samples:
            return _fallback_dailydialog()

    with open(marker, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    print(f"  [DailyDialog] {len(samples):,} utterances -> {marker}")
    return samples


def _parse_dialog_file(filepath, emotion_filepath, emo_map):
    """Parse a plain dialogues_text.txt file on disk."""
    samples = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            dialogs = f.read().strip().splitlines()
        emotions_lines = []
        if emotion_filepath and os.path.exists(emotion_filepath):
            with open(emotion_filepath, "r", encoding="utf-8") as f:
                emotions_lines = f.read().strip().splitlines()
        for i, dialog in enumerate(dialogs):
            turns = [t.strip() for t in dialog.split("__eou__") if t.strip()]
            emos = []
            if i < len(emotions_lines):
                emos = [int(x.strip()) for x in emotions_lines[i].split() if x.strip().isdigit()]
            for j, turn in enumerate(turns):
                emo_label = emo_map.get(emos[j], "neutral") if j < len(emos) else "neutral"
                context = " | ".join(turns[max(0, j - 2) : j]) if j > 0 else ""
                samples.append(
                    {
                        "text": turn,
                        "emotion": emo_label,
                        "context": context,
                        "turn_index": j,
                    }
                )
    except Exception as e:
        print(f"  [DailyDialog] File parse error {filepath}: {e}")
    return samples


def _fallback_dailydialog():
    """Use HuggingFace datasets library as fallback for DailyDialog."""
    marker = os.path.join(TRAINING_DIR, "dailydialog_samples.json")
    print("  [DailyDialog] Trying HuggingFace datasets API...")
    try:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "datasets", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        from datasets import load_dataset

        emo_map = {
            0: "neutral",
            1: "anger",
            2: "disgust",
            3: "fear",
            4: "joy",
            5: "sadness",
            6: "surprise",
        }
        ds = load_dataset("li2017dailydialog/daily_dialog", split="train", trust_remote_code=True)
        samples = []
        for row in ds:
            turns = row.get("dialog", [])
            emotions = row.get("emotion", [])
            for j, turn in enumerate(turns):
                turn = turn.strip()
                if not turn:
                    continue
                emo_label = emo_map.get(emotions[j], "neutral") if j < len(emotions) else "neutral"
                context = " | ".join(turns[max(0, j - 2) : j]) if j > 0 else ""
                samples.append(
                    {
                        "text": turn,
                        "emotion": emo_label,
                        "context": context,
                        "turn_index": j,
                    }
                )
            if len(samples) >= _cap_dailydialog():
                break
        with open(marker, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False)
        print(f"  [DailyDialog] (HuggingFace) {len(samples):,} utterances -> {marker}")
        return samples
    except Exception as e:
        print(f"  [DailyDialog] HuggingFace fallback also failed: {e}")
        if not os.path.exists(marker):
            with open(marker, "w", encoding="utf-8") as f:
                json.dump([], f)
        else:
            print(
                "  [DailyDialog] WARNING: keeping existing file instead of overwriting with empty data"
            )
        return []


# =====================================================================
# IMPORT: Load downloaded data into HaromaX6 modules
# =====================================================================


def _entity_id(name: str, entity_type: str) -> str:
    raw = f"{name.lower().strip()}|{entity_type}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def import_knowledge(triples: list):
    """Import triples into a KnowledgeGraph JSON file that Persistence loads."""
    if not triples:
        return

    print("\n--- Importing into KnowledgeGraph ---")
    entities = {}
    relations = []
    now = time.time()

    for t in triples:
        subj = t["subject"]
        pred = t["predicate"]
        obj = t["object"]
        conf = t.get("confidence", 0.8)
        source = t.get("source", "import")

        for name in (subj, obj):
            eid = _entity_id(name, "CONCEPT")
            if eid not in entities:
                entities[eid] = {
                    "id": eid,
                    "name": name,
                    "entity_type": "CONCEPT",
                    "properties": {},
                    "first_seen": now,
                    "last_seen": now,
                    "mention_count": 1,
                }
            else:
                entities[eid]["mention_count"] += 1
                entities[eid]["last_seen"] = now

        relations.append(
            {
                "subject_id": _entity_id(subj, "CONCEPT"),
                "predicate": pred,
                "object_id": _entity_id(obj, "CONCEPT"),
                "confidence": conf,
                "timestamp": now,
                "source_cycle": 0,
                "source": source,
            }
        )

    kg_data = {
        "entities": list(entities.values()),
        "relations": relations,
    }

    out = os.path.join(COGNITIVE_DIR, "knowledge_graph.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(kg_data, f, ensure_ascii=False)
    print(f"  {len(entities):,} entities, {len(relations):,} relations -> {out}")


def import_emotion_lexicon(emolex: dict):
    """Import EmoLex into the NLU LearnedLexicon persistence format.

    LearnedLexicon.from_dict expects:
      {"entries": {word: float_valence}, "observation_counts": {word: int}}
    """
    if not emolex:
        return

    print("\n--- Importing Emotion Lexicon ---")

    # Map NRC emotion categories to a single valence score
    emo_valence = {
        "positive": 0.5,
        "joy": 0.8,
        "trust": 0.4,
        "anticipation": 0.3,
        "surprise": 0.1,
        "negative": -0.5,
        "anger": -0.7,
        "fear": -0.6,
        "sadness": -0.7,
        "disgust": -0.8,
    }

    entries = {}
    observation_counts = {}
    for word, emotions in emolex.items():
        val_scores = [emo_valence.get(e, 0.0) for e in emotions]
        valence = sum(val_scores) / len(val_scores) if val_scores else 0.0
        entries[word] = round(valence, 3)
        observation_counts[word] = 10

    lexicon_data = {
        "entries": entries,
        "observation_counts": observation_counts,
        "learned_count": len(entries),
        "updated_count": 0,
    }

    out = os.path.join(COGNITIVE_DIR, "nlu_lexicon.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(lexicon_data, f, ensure_ascii=False)
    print(f"  {len(entries):,} words -> {out}")

    # Also save the full emotion-tagged version for training reference
    full_out = os.path.join(TRAINING_DIR, "emolex_full.json")
    full_lex = {}
    for word, emotions in emolex.items():
        val_scores = [emo_valence.get(e, 0.0) for e in emotions]
        valence = sum(val_scores) / len(val_scores) if val_scores else 0.0
        full_lex[word] = {
            "valence": round(valence, 3),
            "emotions": emotions,
        }
    with open(full_out, "w", encoding="utf-8") as f:
        json.dump(full_lex, f, ensure_ascii=False)
    print(f"  Full emotion tags -> {full_out}")


def import_vocabulary(words: list, dialog_samples: list):
    """Import frequency words + dialog vocabulary into LanguageComposer format."""
    print("\n--- Importing Vocabulary ---")

    all_words = set(words)

    # Extract words from dialog samples
    if dialog_samples:
        for sample in dialog_samples[: _cap_vocab_dialog_rows()]:
            text = sample.get("text", "")
            for w in text.lower().split():
                cleaned = "".join(c for c in w if c.isalpha())
                if cleaned and len(cleaned) > 1:
                    all_words.add(cleaned)

    vocab_list = sorted(all_words)[:4000]

    specials = ["<pad>", "<sos>", "<eos>", "<unk>"]
    full_vocab = specials + [w for w in vocab_list if w not in specials]

    out = os.path.join(COGNITIVE_DIR, "word_vocabulary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"words": full_vocab}, f, ensure_ascii=False)
    print(f"  {len(full_vocab):,} words (incl. specials) -> {out}")


def import_dialog_training(samples: list):
    """Save dialog samples as training data for LanguageComposer."""
    if not samples:
        return

    print("\n--- Importing Dialog Training Data ---")

    training = []
    for s in samples:
        training.append(
            {
                "text": s["text"],
                "emotion": s.get("emotion", "neutral"),
                "context": s.get("context", ""),
            }
        )

    out = os.path.join(TRAINING_DIR, "dialog_training.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(training, f, ensure_ascii=False)
    print(f"  {len(training):,} utterances -> {out}")


# =====================================================================
# Main
# =====================================================================


def main():
    global _DOWNLOAD_QUICK
    import argparse

    ap = argparse.ArgumentParser(description="Download public data and import into HaromaX6 data/")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Smaller ConceptNet/WordNet/DailyDialog caps for faster setup",
    )
    args = ap.parse_args()
    _DOWNLOAD_QUICK = args.quick

    print("=" * 60)
    print("  HaromaX6 Training Data Downloader & Importer")
    if _DOWNLOAD_QUICK:
        print("  (quick mode: reduced corpus sizes)")
    print("=" * 60)

    # Download
    cn_triples = download_conceptnet()
    wn_triples = download_wordnet()
    emolex = download_emolex()
    freq_words = download_word_freq()
    dialog = download_dailydialog()

    # Import
    all_triples = cn_triples + wn_triples
    import_knowledge(all_triples)
    import_emotion_lexicon(emolex)
    import_vocabulary(freq_words, dialog)
    import_dialog_training(dialog)

    print("\n" + "=" * 60)
    print("  Download & Import Complete!")
    print("=" * 60)
    print(f"\n  Knowledge:    {len(all_triples):>8,} triples")
    print(f"  Emotions:     {len(emolex):>8,} words")
    print(f"  Vocabulary:   {len(freq_words):>8,} base words")
    print(f"  Dialogues:    {len(dialog):>8,} utterances")
    print(f"\n  Data dir:     {DATA_DIR}")
    print(f"  Training dir: {TRAINING_DIR}")
    print(f"  Cognitive:    {COGNITIVE_DIR}")
    print()


if __name__ == "__main__":
    main()
