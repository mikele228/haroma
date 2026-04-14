"""
NLUProcessor — Natural Language Understanding for HaromaX6
(Phase 5 → Phase 11).

Uses spaCy to extract structured linguistic information from text:
  - Named entities (PERSON, ORG, GPE, CONCEPT, ...)
  - Subject-Verb-Object relation triples from dependency parse
  - Coarse modality label (utterance / imperative / exclamatory — no ? / WH split)
  - Sentiment polarity via emotion-keyword lexicon with negation handling
  - Coreference hints via noun-chunk recency heuristic

Phase 11 upgrade: The static _EMOTION_LEXICON is replaced by a
LearnedLexicon that starts from the same seed words but grows from
experience — learning new word→polarity associations from emotional
context and updating existing ones via running-average feedback.
"""

from typing import Dict, Any, List, Optional, Tuple
import re

try:
    import spacy

    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None

_SEED_LEXICON: Dict[str, float] = {
    "happy": 0.8,
    "joy": 0.9,
    "delight": 0.8,
    "wonderful": 0.7,
    "beautiful": 0.6,
    "love": 0.8,
    "warm": 0.4,
    "bright": 0.3,
    "success": 0.6,
    "wonder": 0.7,
    "awe": 0.6,
    "amazing": 0.7,
    "incredible": 0.6,
    "discover": 0.5,
    "revelation": 0.5,
    "curious": 0.3,
    "explore": 0.3,
    "peace": 0.5,
    "calm": 0.4,
    "serene": 0.5,
    "harmony": 0.5,
    "resolve": 0.3,
    "determined": 0.3,
    "fear": -0.7,
    "afraid": -0.7,
    "dark": -0.3,
    "danger": -0.6,
    "threat": -0.6,
    "terror": -0.9,
    "dread": -0.8,
    "anxious": -0.5,
    "sad": -0.7,
    "loss": -0.5,
    "grief": -0.8,
    "alone": -0.4,
    "empty": -0.4,
    "gone": -0.2,
    "miss": -0.3,
    "tears": -0.5,
    "anger": -0.6,
    "angry": -0.7,
    "furious": -0.8,
    "injustice": -0.5,
    "wrong": -0.3,
    "rage": -0.8,
    "betray": -0.7,
    "surprise": 0.1,
    "unexpected": 0.0,
    "shock": -0.2,
    "good": 0.4,
    "great": 0.5,
    "bad": -0.4,
    "terrible": -0.7,
    "nice": 0.3,
    "horrible": -0.7,
    "hate": -0.8,
    "like": 0.3,
    "enjoy": 0.5,
    "hurt": -0.5,
    "pain": -0.6,
    "help": 0.3,
    "kind": 0.4,
    "cruel": -0.6,
    "gentle": 0.4,
    "harsh": -0.4,
}

_NEGATION_WORDS = frozenset(
    {
        "not",
        "no",
        "never",
        "neither",
        "nobody",
        "nothing",
        "nowhere",
        "nor",
        "cannot",
        "n't",
        "dont",
        "doesn",
        "didn",
        "won",
        "wouldn",
        "couldn",
        "shouldn",
    }
)

_STRIP_RE = re.compile(r"[^a-z0-9' ]+")

_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "because",
        "but",
        "and",
        "or",
        "if",
        "while",
        "about",
        "against",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "us",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "it",
        "its",
        "they",
        "them",
        "what",
        "which",
        "who",
        "whom",
        "up",
        "down",
    }
)


class LearnedLexicon:
    """Growing emotion lexicon that learns from experiential context."""

    _LEARNING_RATE = 0.1
    _MIN_OBSERVATIONS = 3
    _MAX_ENTRIES = 2000

    def __init__(self):
        self._entries: Dict[str, float] = dict(_SEED_LEXICON)
        self._observation_counts: Dict[str, int] = {w: 100 for w in _SEED_LEXICON}
        self._learned_count = 0
        self._updated_count = 0

    def get(self, word: str) -> Optional[float]:
        return self._entries.get(word)

    def contains(self, word: str) -> bool:
        return word in self._entries

    def learn_from_context(
        self, words: List[str], context_polarity: float, confidence: float = 1.0
    ):
        if abs(context_polarity) < 0.05:
            return

        for word in words:
            if word in _STOP_WORDS or word in _NEGATION_WORDS:
                continue
            if len(word) < 2 or not word.isalpha():
                continue

            if word in self._entries:
                count = self._observation_counts.get(word, 1)
                alpha = self._LEARNING_RATE * confidence / (1.0 + count * 0.01)
                old = self._entries[word]
                self._entries[word] = old + alpha * (context_polarity - old)
                self._entries[word] = max(-1.0, min(1.0, self._entries[word]))
                self._observation_counts[word] = count + 1
                self._updated_count += 1
            else:
                if len(self._observation_counts) > self._MAX_ENTRIES * 3:
                    cutoff = sorted(self._observation_counts.values())[
                        len(self._observation_counts) // 2
                    ]
                    self._observation_counts = {
                        k: v
                        for k, v in self._observation_counts.items()
                        if v > cutoff or k in self._entries
                    }
                obs = self._observation_counts.get(word, 0)
                self._observation_counts[word] = obs + 1

                if self._observation_counts[word] >= self._MIN_OBSERVATIONS:
                    if len(self._entries) < self._MAX_ENTRIES:
                        self._entries[word] = max(
                            -1.0, min(1.0, context_polarity * confidence * 0.5)
                        )
                        self._learned_count += 1

    @property
    def total_words(self) -> int:
        return len(self._entries)

    @property
    def seed_size(self) -> int:
        return len(_SEED_LEXICON)

    @property
    def grown_words(self) -> int:
        return self.total_words - self.seed_size

    def stats(self) -> Dict[str, Any]:
        return {
            "total_words": self.total_words,
            "seed_size": self.seed_size,
            "grown_words": self.grown_words,
            "learned_new": self._learned_count,
            "updates": self._updated_count,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": dict(self._entries),
            "observation_counts": dict(self._observation_counts),
            "learned_count": self._learned_count,
            "updated_count": self._updated_count,
        }

    def from_dict(self, data: Dict[str, Any]):
        saved_entries = data.get("entries")
        if saved_entries and isinstance(saved_entries, dict):
            self._entries = dict(saved_entries)
        saved_counts = data.get("observation_counts")
        if saved_counts and isinstance(saved_counts, dict):
            try:
                self._observation_counts = {k: int(v) for k, v in saved_counts.items()}
            except (TypeError, ValueError):
                self._observation_counts = {}
        self._learned_count = data.get("learned_count", 0)
        self._updated_count = data.get("updated_count", 0)


class NLUProcessor:
    """Extracts structured linguistic features from text using spaCy."""

    def __init__(self):
        self._recent_nouns: List[str] = []
        self._max_recent = 20
        self.lexicon = LearnedLexicon()

    def process(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return self._empty_result()

        if _nlp is None:
            return self._fallback(text)

        doc = _nlp(text)

        entities = self._extract_entities(doc)
        relations = self._extract_relations(doc)
        intent = self._classify_intent(doc, text)
        sentiment = self._compute_sentiment(doc)
        noun_phrases = self._extract_noun_phrases(doc)
        root_verb = self._find_root_verb(doc)
        negated = self._is_negated(doc)
        coref_resolved = self._resolve_coreference(doc, entities)
        aux_verbs = self._extract_aux_verbs(doc)
        subordinate_clauses = self._extract_subordinate_clauses(doc)

        self._update_recent_nouns(noun_phrases)

        return {
            "entities": entities,
            "relations": relations,
            "intent": intent,
            "sentiment": sentiment,
            "noun_phrases": noun_phrases,
            "root_verb": root_verb,
            "negated": negated,
            "coref_entities": coref_resolved,
            "aux_verbs": aux_verbs,
            "subordinate_clauses": subordinate_clauses,
        }

    def learn_from_emotion(self, text: str, emotion_valence: float, confidence: float = 1.0):
        if not text or abs(emotion_valence) < 0.05:
            return
        clean = _STRIP_RE.sub(" ", text.lower())
        words = [w for w in clean.split() if len(w) >= 2]
        self.lexicon.learn_from_context(words, emotion_valence, confidence)

    def _extract_entities(self, doc) -> List[Dict[str, Any]]:
        entities: List[Dict[str, Any]] = []
        seen: set = set()

        for ent in doc.ents:
            key = (ent.text.lower(), ent.label_)
            if key in seen:
                continue
            seen.add(key)

            role = "mention"
            for token in ent:
                if token.dep_ in ("nsubj", "nsubjpass"):
                    role = "subject"
                    break
                elif token.dep_ in ("dobj", "pobj", "attr"):
                    role = "object"
                    break

            entities.append(
                {
                    "text": ent.text,
                    "type": ent.label_,
                    "role": role,
                    "start": ent.start_char,
                    "end": ent.end_char,
                }
            )

        for chunk in doc.noun_chunks:
            chunk_lower = chunk.text.lower()
            if any(chunk_lower == e["text"].lower() for e in entities):
                continue
            if chunk.root.pos_ == "PRON":
                continue
            if len(chunk.text.split()) >= 2 or chunk.root.pos_ == "PROPN":
                key = (chunk_lower, "CONCEPT")
                if key not in seen:
                    seen.add(key)
                    role = "subject" if chunk.root.dep_ in ("nsubj", "nsubjpass") else "object"
                    entities.append(
                        {
                            "text": chunk.text,
                            "type": "CONCEPT",
                            "role": role,
                            "start": chunk.start_char,
                            "end": chunk.end_char,
                        }
                    )

        return entities

    def _extract_relations(self, doc) -> List[Dict[str, Any]]:
        relations: List[Dict[str, Any]] = []

        for token in doc:
            if token.pos_ != "VERB":
                continue

            subjects: List[str] = []
            objects: List[str] = []
            aux_verbs: List[str] = []

            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subjects.append(self._get_span_text(child))
                elif child.dep_ in ("dobj", "attr"):
                    objects.append(self._get_span_text(child))
                elif child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            objects.append(self._get_span_text(pobj))
                elif child.dep_ == "aux" or child.dep_ == "auxpass":
                    aux_verbs.append(child.text.lower())

            predicate = token.lemma_
            neg = any(c.dep_ == "neg" for c in token.children)
            if neg:
                predicate = f"not_{predicate}"

            for subj in subjects:
                for obj in objects:
                    relations.append(
                        {
                            "subject": subj,
                            "predicate": predicate,
                            "object": obj,
                            "negated": neg,
                            "verb_pos": token.i,
                            "aux_verbs": aux_verbs,
                        }
                    )

            if not objects and subjects:
                for subj in subjects:
                    relations.append(
                        {
                            "subject": subj,
                            "predicate": predicate,
                            "object": "",
                            "negated": neg,
                            "verb_pos": token.i,
                            "aux_verbs": aux_verbs,
                        }
                    )

        return relations

    def _classify_intent(self, doc, text: str) -> str:
        text_stripped = text.strip()
        if text_stripped.endswith("!"):
            return "exclamatory"

        root = None
        for token in doc:
            if token.dep_ == "ROOT":
                root = token
                break

        if root and root.pos_ == "VERB":
            has_subject = any(c.dep_ in ("nsubj", "nsubjpass") for c in root.children)
            if not has_subject and root.tag_ in ("VB", "VBP"):
                return "imperative"

        return "utterance"

    def _compute_sentiment(self, doc) -> Dict[str, float]:
        polarity_sum = 0.0
        subjectivity_sum = 0.0
        word_count = 0
        negation_scope = 0

        for token in doc:
            lemma = token.lemma_.lower()
            text_lower = token.text.lower()

            if text_lower in _NEGATION_WORDS or lemma in _NEGATION_WORDS:
                negation_scope = 3
                continue

            score = self.lexicon.get(lemma) or self.lexicon.get(text_lower)
            if score is not None:
                if negation_scope > 0:
                    score = -score * 0.8
                polarity_sum += score
                subjectivity_sum += abs(score)
                word_count += 1

            if negation_scope > 0:
                negation_scope -= 1

        if word_count == 0:
            return {"polarity": 0.0, "subjectivity": 0.0}

        polarity = max(-1.0, min(1.0, polarity_sum / word_count))
        subjectivity = min(1.0, subjectivity_sum / max(word_count, 1))
        return {
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
        }

    def _extract_noun_phrases(self, doc) -> List[str]:
        return [chunk.text.lower() for chunk in doc.noun_chunks]

    def _find_root_verb(self, doc) -> str:
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return token.lemma_
        return ""

    def _is_negated(self, doc) -> bool:
        for token in doc:
            if token.dep_ == "ROOT":
                return any(c.dep_ == "neg" for c in token.children)
        return False

    def _resolve_coreference(self, doc, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        resolved: List[Dict[str, Any]] = []
        for token in doc:
            if token.pos_ != "PRON" or token.text.lower() in ("i", "me", "my", "we", "us"):
                continue

            antecedent = self._find_antecedent(token, entities)
            if antecedent:
                resolved.append(
                    {
                        "pronoun": token.text,
                        "antecedent": antecedent,
                        "position": token.i,
                    }
                )
        return resolved

    def _find_antecedent(self, pronoun_token, entities: List[Dict[str, Any]]) -> str:
        if entities:
            for ent in reversed(entities):
                if ent.get("end", 999) < pronoun_token.idx:
                    return ent["text"]
            return entities[-1]["text"]

        if self._recent_nouns:
            return self._recent_nouns[-1]
        return ""

    def _get_span_text(self, token) -> str:
        subtree = list(token.subtree)
        start = subtree[0].i
        end = subtree[-1].i + 1
        return token.doc[start:end].text

    def _update_recent_nouns(self, noun_phrases: List[str]):
        self._recent_nouns.extend(noun_phrases)
        if len(self._recent_nouns) > self._max_recent:
            self._recent_nouns = self._recent_nouns[-self._max_recent :]

    def _extract_aux_verbs(self, doc) -> List[str]:
        """Extract auxiliary verbs from the sentence (can, could, might, etc.)."""
        auxes: List[str] = []
        for token in doc:
            if token.dep_ in ("aux", "auxpass") and token.pos_ == "AUX":
                auxes.append(token.text.lower())
        return auxes

    def _extract_subordinate_clauses(self, doc) -> List[Dict[str, Any]]:
        """Detect clausal complements (ccomp, xcomp, advcl) for nested framing."""
        clauses: List[Dict[str, Any]] = []
        for token in doc:
            if token.dep_ in ("ccomp", "xcomp", "advcl") and token.pos_ == "VERB":
                subj = ""
                obj = ""
                aux_list: List[str] = []
                neg = False
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subj = self._get_span_text(child)
                    elif child.dep_ in ("dobj", "attr"):
                        obj = self._get_span_text(child)
                    elif child.dep_ in ("aux", "auxpass"):
                        aux_list.append(child.text.lower())
                    elif child.dep_ == "neg":
                        neg = True

                clauses.append(
                    {
                        "dep": token.dep_,
                        "verb": token.lemma_,
                        "subject": subj,
                        "object": obj,
                        "aux_verbs": aux_list,
                        "negated": neg,
                    }
                )
        return clauses

    def _fallback(self, text: str) -> Dict[str, Any]:
        clean = _STRIP_RE.sub(" ", text.lower())
        words = clean.split()

        polarity = 0.0
        count = 0
        for w in words:
            score = self.lexicon.get(w)
            if score is not None:
                polarity += score
                count += 1

        intent = "utterance"
        if text.strip().endswith("!"):
            intent = "exclamatory"

        negated = any(w in _NEGATION_WORDS for w in words)

        return {
            "entities": [],
            "relations": [],
            "intent": intent,
            "sentiment": {
                "polarity": round(polarity / max(count, 1), 3) if count else 0.0,
                "subjectivity": round(min(1.0, count / max(len(words), 1)), 3),
            },
            "noun_phrases": [],
            "root_verb": "",
            "negated": negated,
            "coref_entities": [],
            "aux_verbs": [],
            "subordinate_clauses": [],
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "entities": [],
            "relations": [],
            "intent": "none",
            "sentiment": {"polarity": 0.0, "subjectivity": 0.0},
            "noun_phrases": [],
            "root_verb": "",
            "negated": False,
            "coref_entities": [],
            "aux_verbs": [],
            "subordinate_clauses": [],
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "spacy_loaded": _nlp is not None,
            "recent_nouns": len(self._recent_nouns),
            "lexicon": self.lexicon.stats(),
        }
