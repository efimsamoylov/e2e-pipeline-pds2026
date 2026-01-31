## 🤖 Model Weights

The model weights are stored on Google Drive due to their size.  

👉 **[Download Model Checkpoint (Google Drive)](https://drive.google.com/drive/folders/1hxRcc6ispLhaqtyVhdRws1gtHEBO2DHO?usp=share_link)**

### Installation
1. Download the archive from the link above.
2. Extract it to `e2e_pipline/`.
3. Run the pipeline.

# **Baseline**

This script implements a strict, rule-based baseline algorithm for classifying job titles. It serves as a benchmark (lower bound) to evaluate the performance of more complex models.

#### **Algorithm Logic**

1.  **Strict Matching (No Fallback):**
    *   The algorithm attempts to predict the **Department** and **Seniority** of a person based on their job title.
    *   It uses pre-defined lookup tables (dictionaries) created from CSV files.
    *   A prediction is considered valid **only if** there is an exact match (after normalization) between the profile's job title and an entry in the dictionary.
    *   If no match is found, the algorithm returns `None` (it does not guess or use a default "Other" value). This ensures that the accuracy metric reflects pure dictionary coverage rather than random chance.

2.  **Text Normalization:**
    *   To improve matching, all text (both in the dictionary and in the profiles) undergoes a normalization process:
        *   Converted to lowercase.
        *   Non-alphanumeric characters (punctuation, symbols) are removed.
        *   Multiple spaces are collapsed into a single space.
    *   Example: `"Sr. Manager, Sales"` becomes `"sr manager sales"`.

3.  **Smart Job Selection:**
    *   A user profile may contain multiple job entries. The algorithm intelligently selects the most relevant one:
        *   It filters for positions with `status: "ACTIVE"`.
        *   It sorts these active positions by `startDate` in descending order.
        *   It selects the **most recent** active job for classification.

#### **Code Flow**

1.  **Data Loading:**
    *   Loads Department and Seniority mappings from CSV files (`department-v2.csv`, `seniority-v2.csv`).
    *   Loads the ground truth dataset (`linkedin-cvs-annotated.json`) containing user profiles with known labels.

2.  **Dictionary Construction:**
    *   Builds two lookup dictionaries (one for Department, one for Seniority) where keys are normalized job titles and values are the corresponding labels.

3.  **Main Loop (Evaluation):**
    *   Iterates through each profile in the dataset.
    *   Extracts the current job title using the selection logic described above.
    *   Normalizes the job title.
    *   Queries the lookup dictionaries.
    *   Compares the prediction with the ground truth label.

4.  **Reporting:**
    *   Calculates and prints the **Accuracy** for both Department and Seniority.
    *   Accuracy is defined as: `(Number of Correct Exact Matches) / (Total Number of Profiles)`.
## Results: 
    * Department accuracy : 0.073 (7.3%)
    * Seniority  accuracy : 0.151 (12.1%)

# E2E Pipeline - the folder you should focus on!

This directory contains the **End-to-End (E2E) Pipeline** for classifying professional profiles. It serves as the unified production-ready solution, consolidating previous experimental approaches into a modular architecture.

The pipeline is designed to predict:
1.  **Department** (e.g., Engineering, Sales, HR)
2.  **Seniority** (e.g., Junior, Senior, C-Level)

## 🧠 Algorithms & Approaches

This pipeline supports two distinct classification strategies, located in `src/algorithms/`:

### 1. Hybrid Lexicon (Recommended)
**Location:** `src/algorithms/hybrid/`  
**Logic:** A "Smart" combination of Rule-Based logic and Machine Learning.
*   **Workflow:**
    1.  The system first attempts to classify the text using **Rule-Based Lexicons**.
    2.  If the rule-based confidence is high, that result is used (ensuring 100% precision for known terms).
    3.  If the rule-based result is "Unknown" or low confidence, the text is passed to a **SetFit Model**.
*   **Tech Stack:** `SetFit` (Sentence Transformer Fine-tuning), `pandas`, `scikit-learn`.

### 2. Rule-Based
**Location:** `src/algorithms/rule_based/`  
**Logic:** A deterministic approach based strictly on dictionary matching and scoring.
*   **Workflow:** Calculates scores based on unigrams and bigrams found in the job title. It calculates a "confidence" score based on the margin between the best match and the second-best match.
*   **Use Case:** High speed, interpretability, and scenarios where ML inference is not possible.

---

## 📂 Project Structure

### `src/` (Source Code)
Contains the core logic, split into reusable modules.

*   **`src/algorithms/`**: Implementations of the classification strategies.
    *   **`hybrid/inference.py`**: The main engine for the Hybrid approach. Loads `department_model` and `seniority_model` (SetFit), loads lexicons, and runs `predict_hybrid_smart`.
    *   **`rule_based/inference.py`**: The engine for the dictionary-only approach. Implements scoring logic (`bigram_weight`, `unigram_weight`) and debug scoring.
*   **`src/common/`**: Shared utility functions used by all algorithms.
    *   **`io.py`**: Handles loading JSON profiles, reading Lexicon CSVs, and saving results.
    *   **`text.py`**: Text normalization utilities (cleaning job titles).
    *   **`current_job.py`**: Logic to determine which job in a profile's history is the "current" or relevant one to classify.

### `config/`
Contains Python configuration files defining paths, constants, and thresholds.
*   **`hybrid_lexicon.py`**: Configs for the Hybrid approach (paths to SetFit checkpoints, confidence thresholds).
*   **`rule_based.py`**: Configs for the Rule-Based approach (weights for bigrams/unigrams, default labels).

### `pipelines/`
Entry points for running specific tasks. Scripts here orchestrate the calls to `src/algorithms`.
*   **`run_inference.py`**: for predictions on not-annotated datasets
*   **`run_validation.py`**: for validation on annotated datasets
*   **`pipline.py`**: does a combo of prediction and validation 
*   **`interactive.py`**: is used for single input of the role and recive output as a prediction of department and seniority using the pipeline. 


### `models/`
Storage for the heavy ML model weights.
*   *Note: This folder is empty by default and requires manual download (see below).*

---

## 🤖 Model Weights

The SetFit model weights are required for the **Hybrid** approach. They are stored externally due to their size.

👉 **[Download Model Checkpoint (Google Drive)](https://drive.google.com/drive/folders/1hxRcc6ispLhaqtyVhdRws1gtHEBO2DHO?usp=share_link)**

### Installation
1.  Download the archive from the link above.
2.  Extract the contents.
3.  Ensure the folder structure looks like this:
    ```text
    e2e_pipline/models/
    ├── checkpoint-23570/ 
    └── checkpoints/ 
    ```

---

## 🚀 Usage

To run the inference, use the pipeline scripts configured in your run configurations.

**Example workflow (Hybrid):**
The `inference.py` script generally performs the following steps:
1.  Loads input profiles (`data/*.json`).
2.  Selects the current job for every profile.
3.  Normalizes the job title.
4.  Predicts Department and Seniority using the loaded Lexicons + SetFit models.
5.  Saves the result to a CSV file (e.g., `predictions.csv`).

**Output Columns:**
*   `department_pred`: The predicted class.
*   `department_conf`: Confidence score (0.0 - 1.0).
*   `department_source`: How the decision was made (e.g., "Lexicon", "Model", "Rule").

---

## 📜 History & Legacy Code

The project evolved through several iterations before reaching the final `e2e_pipline`. The root directory contains historical folders representing different developmental stages of our algorithms.

### Rule-Based Approaches
The folders `Rule-based_old` and `Rule-Based` implement the same deterministic, dictionary-matching algorithm. However, they represent vastly different stages of code maturity.

#### 1. `Rule-based_old/` (The Sandbox)
*   **Status:** Deprecated / Experimental.
*   **Description:** This was the initial workspace where the lexicon-based approach was first developed. It contains the raw scripts used to generate the original lexicons using TF-IDF and regex patterns.
*   **Characteristics:**
    *   **No Structure:** A flat collection of scripts (e.g., `build_department_lexicon.py`, `run_rule_based_baseline.py`).
    *   **Hardcoded:** Paths and parameters are often embedded directly in the code.
    *   **Purpose:** Use this folder only to reference the original mathematical logic behind lexicon generation.

#### 2. `Rule-Based/` (The Structured Prototype)
*   **Status:** Legacy (Superseded by `e2e_pipline`).
*   **Description:** This is the **refactored version** of the "old" folder. It takes the exact same algorithmic logic but wraps it in a professional software engineering structure.
*   **Key Improvements:**
    *   **Modular Architecture:** Separation of concerns into `config.py` (settings), `model.py` (core logic), and `inference.py` (execution).
    *   **Reproducibility:** A clear pipeline structure that made it easier to test and debug.
    *   **Significance:** This folder served as the direct architectural blueprint for the final `e2e_pipline`.

    
### Machine Learning Experiments
These folders represent our transition from pure logic to probabilistic models.

#### 3. `Approach 6/` (Classic ML Era)
*   **Tech Stack:** `TF-IDF` + `Logistic Regression`.
*   **Description:** This was our first serious attempt to implement Machine Learning.
    *   It focused on statistical text analysis using N-grams.
    *   **Key Challenge:** We encountered a massive "Domain Shift" problem. The model achieved 97% accuracy on clean training CSVs but dropped to ~23% on real-world messy JSON data.
    *   **Innovation:** We introduced **SMOTE** (Synthetic Minority Over-sampling Technique) here to handle class imbalance and wrote the first iteration of "Quick Win" rules to fix the low recall on IT and Sales roles.

#### 4. `Approach 6_updated/` (The Transformer Pivot)
*   **Tech Stack:** `SetFit` (Sentence Transformers) + `Hybrid Logic`.
*   **Description:** This folder marks the turning point where we abandoned TF-IDF in favor of Deep Learning.
    *   **The Big Change:** Instead of counting words (TF-IDF), we switched to **Semantic Embeddings** using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
    *   **Significance:** This is the direct prototype of the final `e2e_pipline`. It established the **Hybrid Architecture**:
        1.  Check strict rules (100% precision).
        2.  If unknown, ask the SetFit model.
        3.  If model confidence is low, fallback to "Other".


#### 5. `Approach 7_Hybrid_Lexicon/` (The Refinement)
*   **Tech Stack:** `SetFit` + `Dynamic JSON Lexicons`.
*   **Description:** This folder represents the optimization step before the final pipeline.
*   **Key Difference vs Approach 6:**
    *   **Code vs. Data Separation:** In *Approach 6*, the rules were hardcoded in Python (e.g., `if 'developer' in text: ...`). In *Approach 7*, we moved all logic into external JSON files (`department_lexicon.json`).
    *   **Scoring Engine:** Instead of simple `if/else` checks, we reintroduced the robust scoring system (Unigrams/Bigrams) from the *Rule-Based* approach but integrated it into the Machine Learning pipeline.
    *   **Model Reuse:** This approach does not train a new model; it efficiently loads the checkpoints created in `Approach 6_updated`, demonstrating how to decouple training from inference.

#### 🔄 Why we moved from `Approach 6` to `Approach 7`?

While `Approach 6_updated` successfully introduced the SetFit model, it revealed a significant architectural flaw in how we handled **Rules**:

1.  **The "Hardcoding" Problem:**
    *   **In Approach 6:** Rules were embedded directly into the Python code (`rules.py`).
        *   *Example:* `it_keywords = ['software', 'developer', 'java'...]`
    *   **The Issue:** To add a new job title or keyword, we had to modify the source code. As the list grew to hundreds of terms, `rules.py` became unreadable and unmaintainable.
    *   **The Fix (Approach 7):** We extracted all keywords into **External JSON Lexicons**. The code now acts as a generic *engine* that processes data, rather than containing the data itself.

2.  **Boolean vs. Scored Matching:**
    *   **In Approach 6:** The logic was binary.
        *   *Logic:* `if "manager" in text: return "Lead"`
    *   **The Issue:** This lacked nuance. A "Senior Project Manager" contains both "Senior" and "Manager". Which rule wins? In Approach 6, it depended on the order of `if` statements.
    *   **The Fix (Approach 7):** We reintroduced the **Scoring System** from the legacy rule-based approaches.
        *   *Logic:* Bigrams (e.g., "Senior Manager") get higher weights (3.0) than unigrams (e.g., "Manager" = 1.0). The class with the highest cumulative score wins.

3.  **Scalability:**
    *   `Approach 7` allows non-developers to improve the system simply by editing a JSON file, without risking breaking the Python inference logic.


---

## 🏆 Conclusion

This repository documents the complete R&D lifecycle of a classification system.

We moved from:
1.  **Simple Scripts** (`Rule-based_old`) →
2.  **Structured Engineering** (`Rule-Based`) →
3.  **Classical NLP** (`Approach 6`) →
4.  **Deep Learning** (`Approach 6_updated`) →
5.  **Data-Driven Logic** (`Approach 7`).

**The Final Result:**
The `e2e_pipline` is the culmination of this journey. It implements a **Hybrid Architecture** that offers the "best of both worlds":
*   **Precision:** Guaranteed 100% accuracy for known terms via the Lexicon Engine.
*   **Recall:** High-quality generalization for unknown titles via the SetFit Transformer.
*   **Maintainability:** Business logic is decoupled from code, allowing for easy updates without retraining models.
