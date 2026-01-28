This script implements a rule-based baseline algorithm for classifying job titles. It serves as a benchmark (lower bound) to evaluate the performance of more complex models.

#### **Algorithm Logic**

1.  **Strict Matching :**
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
    *   Calculates and prints the Accuracy for both Department and Seniority.
    *   Accuracy is defined as: `(Number of Correct Exact Matches) / (Total Number of Profiles)`.
