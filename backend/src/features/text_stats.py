import re
import pandas as pd
from typing import List

def extract_text_stats(texts: List[str]) -> pd.DataFrame:
    rows = []
    for text in texts:
        t = str(text)
        words = t.split()
        sentences = re.split(r"[.!?]+", t)
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        rows.append({
            "text_length":       len(t),
            "word_count":        len(words),
            "avg_word_length":   round(avg_word_len, 3),
            "sentence_count":    len([s for s in sentences if s.strip()]),
            "exclamation_count": t.count("!"),
            "question_count":    t.count("?"),
        })
    return pd.DataFrame(rows)
