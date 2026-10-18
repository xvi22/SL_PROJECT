# Live Sign Language Recognition (ASL) - Webcam, MediaPipe + LSTM

University group project (team of 5): a **live** system that recognises American Sign Language (ASL) words from an ordinary webcam and shows the word on screen. The signer holds still for about half a second after a sign; the recorded movement goes to the model.

- **Approach:** MediaPipe hand and body landmarks -> cleaning and preprocessing -> LSTM classifier (3-model ensemble)
- **Vocabulary:** the 100 words of WLASL100
- **Training data:** WLASL (Kaggle mirror) + ASL Citizen (Microsoft Research), training only; validation and test are WLASL only

## Results (one-time tests, same test clips)

| Model | Training clips | Top-1, 80 v1 words (85 clips) | Top-1, all 100 words (105 clips) |
|---|---|---|---|
| v1 (WLASL only, 80 words) | 602 | 49.4% | - |
| v2-matched (+ ASL Citizen, same size as the official WLASL100 training split) | 1,426 | 62.4% | 62.9% |
| v2-full (+ all kept ASL Citizen clips) | 3,913 | 69.4% | 70.5% |
| **best-matched** (wider LSTM + handshape features) | 1,426 | **78.8%** | **76.2%** |
| **best-full** (wider LSTM + handshape features) | 3,913 | **78.8%** | **79.0%** |

Published WLASL100 baselines (official test set, not ours): I3D 65.89%, SPOTER 63.18% top-1. Our comparison is indicative only: different, small test set (about +/-9 points).

Best model: 128-unit bidirectional LSTM (1 layer), 390 features per frame (pose, hands relative to the wrist, movement, 44 handshape features), 64 frames, 3 seeds averaged. About 0.56 M parameters per model, about 5 ms per sign for the ensemble on a laptop CPU.

## Repository layout

```
scripts/
  week1_word_selection/   choosing the words from WLASL
  week2_extraction/       MediaPipe landmark extraction and hand cleaning (extract_all_v2.py, clean_hands.py)
  week3_training/         preprocessing, split, training rounds (tune*.py), tests, analysis
  v2_asl_citizen/         ASL Citizen download, extraction, checks, sign-version choice, v2 training sets
  week5_live/             live system code (left-handed mode: mirror_frame.py)
  tools/                  project checks
runs/                     trained models and results
  tuning_v4/              v1 models (E7)
  tuning_v5/              v2 models (w100, matched, full)
  tuning_v8/              BEST models (matched_H128_HAND, full_H128_HAND)
  final_test*/            one-time test results (do not re-run)
report_outputs/           text output of every script used in the report (A*, B*, V2_*)
models/                   MediaPipe model files (hand_landmarker.task, pose_landmarker_full.task)
requirements.txt          Python packages
```

Not in this repository (too large, and dataset licences): `data/`, `wlasl-processed/` (videos), `WLASL-master/`, `venv/`.

## Setup (Windows)

1. Python 3.11, then in the project folder:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Run every script **from the main folder**, for example:
   ```
   python -u scripts\week3_training\tune_v8.py quick
   ```
   Most scripts save their output with `> report_outputs\<name>.txt`.

## Getting the data (not included)

- **WLASL:** Kaggle mirror `risangbaskoro/wlasl-processed` -> unpack to `wlasl-processed\` (videos + `WLASL_v0.3.json`). Original project: github.com/dxli94/WLASL.
- **ASL Citizen:** Microsoft Research (non-commercial licence; cite the paper). `scripts\v2_asl_citizen\download_asl_citizen.py` downloads only the videos needed for our 100 words (about 2.4 GB) instead of the full 42.8 GB zip.
- The datasets are used for research/education only and must not be re-shared publicly.

## Main pipeline (order)

1. Landmarks: `week2_extraction\extract_all_v2.py` -> `clean_hands.py` (v3 rule); ASL Citizen: `v2_asl_citizen\extract_asl_citizen.py`
2. Checks: `v2_asl_citizen\check_asl_citizen_data.py`, `check_gaps_and_discards.py`
3. Sign versions: `variant_auto.py` -> `variant_decide.py` -> `data\asl_citizen\variant_final.csv`
4. Split and preprocessing: `split_new20.py`, `preprocess_v2.py`
5. Training sets: `build_v2_sets.py`
6. Training: `week3_training\tune_v5.py` (v2), `tune_v6.py` (model size), `tune_v7.py` (augmentation), `tune_v8.py` (handshape features, best)
7. One-time tests: `evaluate_test_v2.py`, `evaluate_test_best.py` (they refuse to run twice)
8. Analysis: `analyse_val_v2.py` (validation only)

## Rules we followed

- The test set is used once per model version, after the models are frozen; all choices are made on validation.
- One change at a time, 3 seeds, compare averages and ensembles.
- Every data and model version is kept (new names, nothing overwritten) as evidence.
- Live frames must go through the same cleaning and preprocessing as training (unmirrored camera image to MediaPipe).

## Team

Joshua, Geremy, Zhen Ye, Alicia, Gen Yen
