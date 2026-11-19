# Music Genre Classification: CNN vs. BiLSTM vs. CRNN

We compare three architectures on log-mel spectrograms to see which classifies music genre best, and which generalizes best to a dataset it has not seen.

- **CNN:** treats the spectrogram as a single-channel image.
- **BiLSTM:** treats the spectrogram as a sequence of frames (128 mel values per frame).
- **CRNN:** conv blocks extract local features, then a BiLSTM models how they change over time.

**Core question:** Does treating a spectrogram as an image or as a sequence work better, and does combining them beat both?

| | Dataset | Details |
|---|---|---|
| Train / val / test | [fma_small](https://github.com/mdeff/fma) | 8,000 clips, 8 balanced genres, official split (6,400 / 800 / 800) |
| Cross-dataset test | [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) | Only the 3 overlapping genres (Hip-Hop, Pop, Rock), 300 clips |

**Controls:** same splits, same preprocessing, roughly matched parameter counts, same optimizer / scheduler / batch size / early-stopping rule, same augmentation (SpecAugment time + frequency masking, random gain), fixed seeds.

> Code lives in `src/`; notebooks are thin wrappers. Edit locally, push, then `git pull` in Colab. Do not edit code inside Colab.

---

## Setup

### 1. Environment

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

You also need `ffmpeg` for MP3 decoding.
On Colab: `apt-get install -y ffmpeg`.

### 2. Get the data (not stored in git)

**FMA** (from the [FMA repo](https://github.com/mdeff/fma)):

```bash
mkdir -p data/raw && cd data/raw
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
unzip -q fma_small.zip
unzip -q fma_metadata.zip
```

**GTZAN** (Kaggle; requires a Kaggle API token in `~/.kaggle/kaggle.json`, never commit it):

```bash
kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification
unzip -q gtzan-dataset-music-genre-classification.zip -d gtzan
```

**On Colab:** unzip to local disk (`/content/data/raw`), *not* Drive. Only write processed outputs to Drive.