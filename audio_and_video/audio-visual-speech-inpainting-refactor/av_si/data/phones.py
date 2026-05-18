"""GRID alignment → phone sequence (port of transcription2phonemes.py)."""
from pathlib import Path
from typing import List


def load_phone_dictionary(path: str) -> List[str]:
    """Read TIMIT phone dictionary, return sorted unique phones (drops empty/SP)."""
    text = Path(path).read_text()
    phones = sorted({p for p in text.replace("\n", " ").split(" ") if p and p != "SP"})
    return phones


def load_word_dictionary(word_file: str, dict_file: str):
    words = Path(word_file).read_text().splitlines()
    dicts = Path(dict_file).read_text().splitlines()
    return words, dicts


def words_to_phones(transcription: str, words: List[str], dicts: List[str]) -> str:
    out = transcription
    for w, d in zip(words, dicts):
        out = out.replace(w, d)
    return out


def linearize_phones(phone_text: str) -> List[str]:
    """Strip alignment timings/SIL, return ordered phone list."""
    out = []
    for tok in phone_text.replace("\n", " ").split(" "):
        if tok.isalpha() and tok != "SIL":
            out.append(tok)
    return out


def align_to_phone_labels(align_path: str, words: List[str], dicts: List[str], phone_dict: List[str]):
    """One GRID .align file → list[int] of phone indices into phone_dict."""
    transcription = Path(align_path).read_text()
    phones = linearize_phones(words_to_phones(transcription, words, dicts))
    return [phone_dict.index(p) for p in phones if p in phone_dict]
