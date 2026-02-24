import logging
import pandas as pd
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def preprocess_for_lda(texts: List[str]) -> List[List[str]]:
    import re, nltk
    try: nltk.data.find("corpora/stopwords")
    except LookupError: nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    stop = set(stopwords.words("english"))
    stop.update(["said","also","would","could","one","two","may","new"])
    result = []
    for text in texts:
        text = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
        tokens = [w for w in text.split() if len(w) > 2 and w not in stop]
        result.append(tokens)
    return result

def train_lda(texts: List[str], num_topics: int = 10, passes: int = 10, save_path: Optional[Path] = None):
    from gensim import corpora
    from gensim.models import LdaModel
    processed  = preprocess_for_lda(texts)
    dictionary = corpora.Dictionary(processed)
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    corpus = [dictionary.doc2bow(doc) for doc in processed]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes, random_state=42, alpha="auto")
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        lda.save(str(save_path / "lda_model"))
        dictionary.save(str(save_path / "lda_dictionary"))
    return lda, dictionary

def extract_topic_features(texts: List[str], lda_model=None, dictionary=None,
                            num_topics: int = 10, model_dir: Optional[Path] = None) -> pd.DataFrame:
    from gensim.models import LdaModel
    from gensim import corpora
    if lda_model is None and model_dir is not None:
        try:
            lda_model  = LdaModel.load(str(model_dir / "lda_model"))
            dictionary = corpora.Dictionary.load(str(model_dir / "lda_dictionary"))
        except:
            lda_model, dictionary = train_lda(texts, num_topics)
    elif lda_model is None:
        lda_model, dictionary = train_lda(texts, num_topics)
    processed = preprocess_for_lda(texts)
    rows = []
    for doc in processed:
        bow = dictionary.doc2bow(doc)
        dist = dict(lda_model.get_document_topics(bow, minimum_probability=0.0))
        rows.append({f"topic_{i}": dist.get(i, 0.0) for i in range(num_topics)})
    return pd.DataFrame(rows)
