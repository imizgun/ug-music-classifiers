# Klasyfikacja nastroju muzycznego — Plan projektu

**Temat:** Klasyfikacja nastroju utworów muzycznych na podstawie cech audio i embeddingów  
**Forma:** Raport badawczy (porównanie modeli) + ewentualnie mini-demo Gradio  
**Cel:** Porównać wiele podejść do klasyfikacji nastroju i wyciągnąć wnioski

---

## 1. Dataset

### Opcja A (rekomendowana): 4Q Audio Emotion Dataset
- 900 piosenek, 4 klasy nastrojów (kwadranty Russella)
- **Q1**: Happy/Excited (wysoka walencja, wysoke pobudzenie)
- **Q2**: Angry/Tense (niska walencja, wysokie pobudzenie)
- **Q3**: Sad/Depressed (niska walencja, niskie pobudzenie)
- **Q4**: Calm/Relaxed (wysoka walencja, niskie pobudzenie)
- Gotowe etykiety, fragmenty 30-sekundowe

### Opcja B: DEAM (Dynamic Emotion in Music)
- 1802 fragmentów, ciągłe wartości valence/arousal (można zbinować)
- Bardziej "naukowy", ale wymaga bucketing etykiet

### Opcja C: Spotify API + własne playlisty
- Pobieramy tracki z playlist "happy", "sad", "energetic", "chill"
- Audio features dostępne przez API (bez pobierania audio)
- Prosto, ale zależność od Spotify

**Wybór: 4Q Dataset** — gotowe etykiety, 4 klasy, dobra wielkość, audio dostępne

---

## 2. Cechy (Features)

### 2a. Hand-crafted (librosa)
| Feature | Opis |
|---|---|
| MFCC (13–40 współczynników) | Barwa dźwięku |
| Spectral centroid | "Jasność" dźwięku |
| Spectral rolloff | Rozkład energii |
| Zero-crossing rate | Szorstkość/noise |
| Chroma features | Harmonia, tonacja |
| Tempo / BPM | Szybkość |
| RMS energy | Głośność |
| Mel spectrogram | Dla CNN |

→ Wyekstrahuj średnią + std z okna czasowego → wektor ~100 cech

### 2b. Embeddings
| Model | Opis |
|---|---|
| **OpenL3** | Embeddingi audio 512-D, pre-trained na AudioSet |
| **MERT** (HuggingFace) | Music-specific transformer embeddings |
| **Essentia TensorFlow models** | Music embeddings z Discogs/MSD |
| Spotify audio features | 11 cech numerycznych (gotowe, przez API) |

### 2c. Reprezentacja obrazowa (dla CNN)
- **Mel spectrogram** → obraz 128×128 px
- **Chromagram**
- Można użyć augmentacji: time stretch, pitch shift, dodanie szumu

---

## 3. Modele do porównania

> 5 podejść — każde zasadniczo inne

| Model | Dane wejściowe | Co odróżnia | Wymaganie kursu |
|---|---|---|---|
| **Random Forest** | hand-crafted librosa features | klasyczny baseline, interpretowalny | Proste klasyfikatory |
| **MLP** | hand-crafted librosa features | ten sam wход, ale sieć → bezpośrednie porównanie z RF | Sieci neuronowe |
| **CNN** | Mel spectrogram (obraz 128×128) | podejście wizualne, bez cech ręcznych | CNN do klasyfikacji obrazów |
| **LSTM** | sekwencja MFCC w czasie | struktura czasowa, nie agregacja | RNN/LSTM |
| **kNN na embeddingach OpenL3/MERT** | embeddingi 512-D | wiedza pre-trained, brak trenowania | Transfer learning |

### Opcjonalnie (jeden bonus do raportu)
- **GA do selekcji cech RF** — pokazuje które cechy librosa są najważniejsze (Algorytmy genetyczne)
- Nie dodajemy PSO, Fuzzy, Apriori jako osobnych modeli — to nie wnosi nowego do porównania nastrojów

---

## 4. Pipeline

```
Audio (.mp3/.wav)
    │
    ├─► [librosa] → hand-crafted features (CSV)
    │
    ├─► [librosa] → Mel spectrograms (PNG/NPY)
    │
    └─► [OpenL3 / MERT] → embeddingi (NPY)
         │
         ▼
[Preprocessing]
  - Normalizacja (StandardScaler / MinMax)
  - Augmentacja (tylko train set)
  - PCA / t-SNE (wizualizacja)
         │
         ▼
[Klasyfikatory]
  RF / MLP / CNN / LSTM / kNN+embeddingi
         │
         ▼
[Ewaluacja]
  - Accuracy, F1, Confusion Matrix
  - Tabela porównawcza
  - Wizualizacje (t-SNE embeddingów, CM)
```

---

## 5. Ewaluacja i porównanie

- Train/Val/Test split: 70/15/15, stratified
- Metryki: Accuracy, F1-macro, confusion matrix
- Wizualizacja t-SNE embeddingów per klasa
- Tabela końcowa wszystkich modeli — kluczowy element raportu
- Analiza błędów: które nastroje są mylone i dlaczego?

---

## 6. Dodatkowe elementy (optional but nice)

- **Reguły asocjacyjne**: Apriori na zdyskretyzowanych cechach audio → "jeśli wysoki BPM i durowa tonacja → happy"
- **Analiza tekstu**: jeśli dataset ma tytuły/teksty — bag of words / sentiment → porównać z audio-only
- **Mini-demo Gradio**: wgraj plik MP3 → model zwraca nastrój (prosto, ~20 linii)

---

## 7. Stos technologiczny

```
Python 3.11
librosa          — ekstrakcja cech audio, spektrogramy
openl3           — embeddingi audio 512-D
transformers     — MERT embeddingi (alternatywa dla openl3)
scikit-learn     — RF, kNN, preprocessing, metryki
keras / pytorch  — MLP, CNN, LSTM
deap             — GA do selekcji cech (opcjonalnie)
matplotlib/seaborn — wykresy, confusion matrix
umap / sklearn   — t-SNE/UMAP wizualizacja embeddingów
gradio           — opcjonalne demo
```

---

## 8. Fazy realizacji

| Faza | Zadania | Czas szacowany |
|---|---|---|
| 1. Setup | Dataset, środowisko, ekstrakcja cech librosa | 3-4h |
| 2. RF + MLP | Baseline + pierwsza sieć, analiza cech | 2-3h |
| 3. CNN | Spektrogramy jako obrazy, augmentacja | 2-3h |
| 4. LSTM + Embeddingi | Sekwencje MFCC, OpenL3/MERT + kNN | 3-4h |
| 5. Analiza i raport | Tabele, t-SNE, wnioski | 2-3h |
| 6. GA (opcjonalnie) | Selekcja cech dla RF | 1-2h |
| 7. Demo (opcjonalnie) | Gradio app | 1-2h |

**Łącznie: ~13-19h**

---

## 9. Kluczowe wnioski (hipotezy do zweryfikowania)

- Embeddingi (OpenL3/MERT) + kNN powinny bić hand-crafted features + RF/MLP
- CNN na spektrogramach ≈ LSTM na sekwencjach (różne podejścia, podobna info)
- Q2 (Angry) i Q1 (Happy) będą mylone (oba mają wysokie pobudzenie)
- Q3 (Sad) i Q4 (Calm) będą mylone (oba mają niskie pobudzenie)
- RF + GA prawdopodobnie wyeliminuje cechy spektralne, zostawi tempo/energię/chroma

---

## Zasoby / linki

- 4Q Dataset: http://mir.dei.uc.pt/downloads.html
- OpenL3: https://github.com/marl/openl3
- MERT: https://huggingface.co/m-a-p/MERT-v1-95M
- librosa docs: https://librosa.org/doc/latest/
- DEAP dataset: http://www.eecs.qmul.ac.uk/mmv/datasets/deap/
