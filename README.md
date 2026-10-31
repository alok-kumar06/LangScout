# LangScout

**Language detection for 17 languages + Hinglish** (Hindi typed in English letters).

`Python` · `scikit-learn` · `Keras` · `Streamlit` · `FastAPI` · `Docker` · `pytest`

<!-- After you save a screenshot as docs/app.png, remove this comment's markers to show it:
![LangScout app](docs/app.png)
-->

LangScout tells you which language a text is written in, explains why, and says **"not sure"** when the input
is too short, unusual, or in an unsupported script.

**Languages:** Arabic, Danish, Dutch, English, French, German, Greek, Hindi, Italian, Kannada, Malayalam,
Portuguese, Russian, Spanish, Swedish, Tamil, Turkish, and Hinglish.

---

## Results

| Measure                                              | Result                         |
|------------------------------------------------------|--------------------------------|
| Test accuracy (1,842 held-out text chunks)           | **98.6%** (macro-F1 **98.8%**) |
| 5-fold cross-validation macro-F1                     | 98.9% ± 0.2                    |
| Everyday sentences (51, outside the training data)   | 51 / 51                        |
| Hinglish sentences (24, informal spellings)          | 24 / 24                        |
| Model size                                           | 1.2 MB                         |

Accuracy grows with text length: about 70% at 5 characters, 95% at 20, and 98% at 80.

| Model (same data and split)                       | Test accuracy |
|---------------------------------------------------|---------------|
| **Char n-gram TF-IDF + Naive Bayes (deployed)**   | **98.6%**     |
| Char n-gram TF-IDF + Linear SVM                   | 97.9%         |
| Character-level CNN (Keras)                       | 97.6%         |
| Word-level SimpleRNN (my first version)           | 92.8%         |

Against existing libraries (`python -m langscout.benchmark`), LangScout scores 98.6% on the test set versus 97.2%
for langid and 94.7% for langdetect. That comparison favours LangScout because it is trained on the same kind of
text as the test set. On the separate everyday sentences, langid ties it at 100%.

---

## Features

- **Hinglish detection** - Hindi in English letters is its own class, trained on transliterated Hindi with random spelling styles.
- **Explanations** - the app highlights which words pushed the decision towards a language.
- **Honest uncertainty** - unsupported scripts, gibberish and low-confidence input are flagged instead of guessed.
- **Mixed text and batch mode** - detect each sentence separately, or upload a CSV.
- **Four ways to use it** - web app, command line, Python library, REST API.

---

## Quick start

Requires Python 3.11 or newer.

```bash
git clone https://github.com/alok-kumar06/LangScout.git
cd LangScout
python -m venv venv
venv\Scripts\activate          # Mac / Linux: source venv/bin/activate
pip install -r requirements.txt
```

**Web app**

```bash
streamlit run app.py
```

**Command line**

```bash
python -m langscout "Ceci est une belle journée"
python -m langscout --segments "I love this city. Me encanta esta ciudad."
```

**Python**

```python
from langscout import LanguageDetector

result = LanguageDetector().detect("यह एक अच्छा दिन है")
print(result.language, result.confidence, result.status)   # Hindi ... ok
```

**REST API** (interactive docs at `http://127.0.0.1:8000/docs`)

```bash
pip install -r requirements-api.txt
uvicorn langscout.api:app --reload
```

Endpoints: `/health`, `/languages`, `/detect`, `/detect/batch`, `/detect/segments`, `/explain`.

**Retrain, test, benchmark**

```bash
pip install -r requirements-dev.txt   # test and benchmark tools
pytest                                # 46 tests
python -m langscout.benchmark         # comparison with langid and langdetect
python -m langscout.train             # about a minute
python -m langscout.train --deep      # adds Keras models (see requirements-deep.txt)
```

The saved model needs scikit-learn 1.8.0 (pinned). With another version, run the training command once.

---

## How it works

1. **Clean the data** - fix two misspelled labels, remove duplicates, conflicting labels and rows whose script contradicts their label.
2. **Split by row first** (70/15/15, stratified), so text from one article never lands in both training and test data.
3. **Chunk long paragraphs** into sentences of at most 200 characters, so training text looks like what people type.
4. **Features and model** - TF-IDF over character 1 to 3-grams and Naive Bayes, chosen on the validation set only.
5. **Predict** - detect the script, keep only the languages that use it, predict, then return *ok*, *uncertain* or *unsupported*.

**Project layout**

```text
app.py              Streamlit app
langscout/          detector, training, API, explanations, benchmark
data/               Kaggle dataset and extra test sentences
models/             trained model
reports/            metrics and plots
tests/              pytest suite
```

---

## Limitations

- Only these 18 classes; other languages are mapped to the closest one unless their script is unsupported.
- Hinglish spellings vary, Roman Urdu looks almost the same, and the training data is machine-made. The 24-sentence check is small.
- Training text is Wikipedia-style, so slang and very short text (under about 10 characters) are less reliable.
- Hindi has only 62 rows in the dataset, so its numbers rest on a small sample.
- The Dockerfile has not been built or tested yet.

---

## Author

**Alok Kumar** · [LinkedIn](https://www.linkedin.com/in/alok-kumar-91b0a829b/) · [GitHub](https://github.com/alok-kumar06)

Dataset: Kaggle "Language Detection". MIT License.
