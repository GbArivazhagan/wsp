"""Response-language identification with lingua (Apache-2.0), restricted to the languages that matter here.

label(text) -> 'sv' | 'en' | 'da' | 'nb' | 'nn' | 'de' | 'zh' | 'other' | 'empty'
We also report the share of sentence-sized chunks in each language, to catch code-switching.
"""
import re
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder

LANGS = {
    Language.SWEDISH: "sv", Language.ENGLISH: "en", Language.DANISH: "da", Language.BOKMAL: "nb",
    Language.NYNORSK: "nn", Language.GERMAN: "de", Language.CHINESE: "zh", Language.FINNISH: "fi",
    Language.DUTCH: "nl", Language.FRENCH: "fr", Language.SPANISH: "es",
}


@lru_cache(maxsize=1)
def _det():
    return LanguageDetectorBuilder.from_languages(*LANGS).with_preloaded_language_models().build()


_THINK = re.compile(r"<think>.*?</think>", re.S)
_CODE = re.compile(r"```.*?```", re.S)


def clean(text: str) -> str:
    text = _THINK.sub(" ", text)
    text = text.split("</think>")[-1]          # unterminated think blocks: keep the part after the last close
    text = _CODE.sub(" ", text)               # code blocks are not natural language
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return text.strip()


def label(text: str) -> str:
    t = clean(text)
    if len(re.findall(r"[^\W\d_]", t)) < 8:
        return "empty"
    lang = _det().detect_language_of(t)
    return LANGS.get(lang, "other") if lang is not None else "other"


def chunk_shares(text: str):
    """Share of sentence chunks (>=25 letters) per language."""
    t = clean(text)
    chunks = [c for c in re.split(r"(?<=[.!?:\n])\s+", t) if len(re.findall(r"[^\W\d_]", c)) >= 25]
    if not chunks:
        return {}
    out = {}
    for c in chunks:
        l = _det().detect_language_of(c)
        k = LANGS.get(l, "other") if l is not None else "other"
        out[k] = out.get(k, 0) + 1
    return {k: v / len(chunks) for k, v in out.items()}


def label_many(texts):
    return [label(t) for t in texts]
