# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** flex

**Team Members:** Chandu S (Team Leader), Manikanta R B, Shreyas S, Pushkar A

**Submission Date:** September 27, 2026

---

## 1. Executive Summary

We developed an end-to-end, memory-optimized entity resolution pipeline to match 1.73M Source 1 entities against ~10M Source 2 and Source 3 records. The architecture decouples quadratic pairwise search into sparse inverted-index candidate blocking followed by a high-precision XGBoost classifier (`tree_method='hist'`) trained on vectorized RapidFuzz lexical and structural features. Optimized specifically for the competition's $F_{0.5}$ metric via validation threshold sweeps, our model achieved a peak validation $F_{0.5}$ score of **0.8888** with **92.76% Precision** and **76.16% Recall**, passing all official validation checks across 1,732,544 test entities.

---

## 2. Methodology

### 2.1 Problem Analysis

Exploratory analysis revealed three primary modeling challenges:

* **Quadratic Scale:** An unconstrained Cartesian cross-product requires $>17$ trillion comparisons, rendering naive pair generation computationally impossible.
* **Corporate Suffix & Punctuation Noise:** Discrepancies were predominantly driven by legal suffixes (e.g., *"Pvt Ltd"*, *"LLC"*, *"Inc."*, *"Corporation"*) and punctuation variations rather than true brand differences.
* **Severe Address Sparsity & Branch Ambiguity:** A substantial portion of records contained missing or incomplete address fields. Multiple branch locations sharing identical company names caused frequent false-positive risks under loose thresholds.

### 2.2 Solution Strategy

We adopted a decoupled, memory-constrained two-stage supervised architecture:

* **Approach Type:** Two-Stage Pipeline (Sparse Inverted Index Blocking + Histogram-Binning Gradient Boosted Classifier).
* **Core Innovation:** Integration of memory-safe batch processing using PyArrow IPC streams, specialized address-missing interaction flags, and dynamic probability thresholding ($p=0.85$) directly aligned to the precision-heavy $F_{0.5}$ metric objective.

---

## 3. Candidate Generation (Blocking)

To eliminate over 99.9% of non-matching pairs while maintaining high ground-truth recall:

* **Blocking keys used:** Inverted index hashing on normalized name character n-gram tokens and address prefix tokens. Source 2 and Source 3 databases were combined and queried using sparse matrix blocking to avoid memory overhead.
* **Candidate pairs generated:**
* Training pairs: **65,920,448** candidate pairs (~840 MB TSV).
* Test pairs: **51,963,564** candidate pairs (~661 MB TSV).


* **How true matches were preserved:** Multiple overlapping token keys ensured that minor typos or partial omissions did not drop true pairs out of the candidate pool, capturing $>95\%$ of all true positive alignments prior to classification.

---

## 4. Matching Model

**Features used:**

* **Name features:**
* `name_fuzz_ratio`: Levenshtein character edit similarity (RapidFuzz C++ engine).
* `name_token_sort`: Token-sorted similarity resilient to word-order permutations.
* `exact_name_match`: Binary flag for identical normalized strings.


* **Address features:**
* `addr_fuzz_ratio`: Full string edit distance on street and geographic components.
* `addr_token_sort`: Token-sorted ratio capturing rearranged address elements.


* **Sparsity & Interaction Flags:**
* `s1_missing_addr`, `cand_missing_addr`, `both_missing_addr`: Explicit missingness indicators that allow tree splits to depend on name similarity when address fields are absent.



**Model type:** XGBoost Classifier with histogram binning (`tree_method='hist'`), trained with `scale_pos_weight` on a negative-downsampled training partition of 20,671,288 pairs to prevent memory exhaustion while retaining all 5.56M positive pairs.

**Threshold selection method:** Empirical line-sweep across prediction probabilities $p \in [0.50, 0.95]$ on an independent 4.1M-pair stratified holdout set. The maximum $F_{0.5}$ score peaked sharply at **$p = 0.85$**, trading modest recall to eliminate false-positive pairings.

---

## 5. Results & Error Analysis

* **$F_{0.5}$ Score (Validation Peak):** **0.8888** (Precision: **92.76%**, Recall: **76.16%** at threshold $0.85$).
* **Validation Metric Progression Across Thresholds:**

| Threshold ($p$) | Precision | Recall | $F_{0.5}$ Score |
| --- | --- | --- | --- |
| 0.50 | 0.8200 | 0.8647 | 0.8285 |
| 0.65 | 0.8788 | 0.8243 | 0.8673 |
| 0.75 | 0.9075 | 0.7955 | 0.8827 |
| **0.85** | **0.9276** | **0.7616** | **0.8888** |
| 0.90 | 0.9399 | 0.7257 | 0.8875 |

* **Common false positives (wrong merges):** Franchise chains and branch businesses sharing identical normalized brand names located in adjacent postal codes when address records were sparsely populated.
* **Common false negatives (missed matches):** Drastic acronym abbreviations (e.g., *"HDFC"* vs. *"Housing Development Finance Corporation"*) that failed token-overlap thresholds during blocking.

---

## 6. Conclusion

By pairing sparse token inverted-index blocking with a precision-tuned XGBoost model, our pipeline solved multi-source business entity resolution across millions of records without exceeding standard consumer hardware constraints. Strict probability thresholding ($p=0.85$) aligned model predictions with the competition's $F_{0.5}$ evaluation metric, delivering **92.76% precision** and achieving an **0.8888 $F_{0.5}$ score** across 1.73M entities.

---

## Appendix

### A. Code Artefacts

The full pipeline is divided into modular, reproducible stages matching the competition directory specifications:

```text
2. Pipeline/
├── 01_blocking_candidate_generation.ipynb   # Builds sparse candidate indices -> candidate_pairs.tsv
├── 02_feature_engineering.ipynb             # Normalizes text & computes RapidFuzz metrics -> training_features.parquet
├── 03_matching_model.ipynb                  # Trains hist XGBoost & determines optimal threshold (0.85)
└── 04_test_inference_and_submission.ipynb   # Batch test scoring -> matching_results.tsv

```

* **Reproduction Steps:**
1. Run `01_blocking_candidate_generation.ipynb` to generate `output/candidate_pairs.tsv` for both train and test partitions.
2. Run `02_feature_engineering.ipynb` to clean corporate legal noise and compute string similarities into Parquet format.
3. Run `03_matching_model.ipynb` to train the XGBoost model and save `xgb_matching_model.json` with `optimal_threshold.txt`.
4. Run `04_test_inference_and_submission.ipynb` to generate `output/matching_results.tsv`.
5. Validate output compliance with `python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --test-dir student_resource/dataset/test --check-ids`.



### B. Additional Results

* **Test Inference Output Breakdown:**
* Total Source 1 test entities evaluated: **1,732,544**
* Entities with high-confidence matches: **1,618,421**
* Entities deliberately left unmatched (empty): **114,123**
* Official Validator Status: **PASS — 0 blocking issues**