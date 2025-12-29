Thesis Title
Risk Prediction of Antipsychotic-Induced Akathisia in Patients with Bipolar Disorder: A Machine Learning Study Based on the openFDA Database

Abstract
Background: Akathisia is a common and highly distressing extrapyramidal symptom (EPS) associated with antipsychotic treatment. It often leads to poor medication adherence and has been linked to an increased risk of suicide. Although newer agents such as brexpiprazole are marketed as having a lower risk of EPS, large-scale real-world evidence remains limited.
Objective: This study aimed to evaluate the differential risk of akathisia between the novel agent brexpiprazole and the conventional agent aripiprazole using the openFDA Adverse Event Reporting System (FAERS), and to develop a machine learning model for individualized risk prediction.
Methods: Bipolar disorder cases reported by healthcare professionals between 2004 and 2025 were included. Disproportionality analysis was first conducted to calculate reporting odds ratios (RORs) for comparing drug-associated risks. Subsequently, an XGBoost-based machine learning model was constructed and validated using 5-fold cross-validation and a temporal holdout approach, with 2025 data reserved as an independent test set.
Results: Statistical analysis demonstrated that brexpiprazole was associated with a significantly higher risk of akathisia than aripiprazole (overall ROR = 0.60; post-2019 ROR = 0.69; p < 0.001). The machine learning model showed good discriminative performance on the 2025 test set (micro-AUROC = 0.7833) and high sensitivity (recall = 70.4%), albeit with low precision (9.5%).
Conclusion: Real-world evidence suggests that the risk of akathisia associated with brexpiprazole may be underestimated. The proposed model, characterized by high recall, may serve as a useful preliminary clinical screening tool.

Chapter 1. Introduction
1.1 Research Background and Motivation
The treatment of bipolar disorder relies on long-term pharmacological management; however, both diagnosis and treatment remain highly challenging (Nierenberg et al., 2023). Antipsychotic medications constitute a cornerstone of therapy, yet their adverse effects can profoundly impair patients’ quality of life. Among these, akathisia is one of the most common and distressing extrapyramidal symptoms (EPS). Clinically, akathisia manifests as an intense sense of inner restlessness, often leading patients to be misdiagnosed as experiencing “worsening bipolar symptoms” or “acute anxiety,” resulting in inappropriate dose escalation. Given that patients with bipolar disorder already have an elevated risk of suicide, inadequately managed akathisia may further precipitate psychological collapse (Miller & Black, 2020).

1.2 Existing Problems and Research Gaps
With advances in psychopharmacology, novel antipsychotic agents such as brexpiprazole have been developed. These drugs typically emphasize receptor selectivity or partial agonist properties, aiming to reduce EPS-related adverse effects (Tarzian et al., 2023). However, large-scale real-world data validating whether brexpiprazole is indeed safer than its predecessor aripiprazole remain scarce.
Moreover, with the rise of precision medicine, individualized adverse event monitoring using biomarkers has been increasingly applied in fields such as immunotherapy (Yan et al., 2025). In contrast, psychiatric research on adverse drug event prediction largely remains at the level of traditional population-based statistics, lacking dynamic, individualized risk prediction models.

1.3 Research Objectives
Using the openFDA database, this study aimed to:
Compare the real-world risk of akathisia associated with aripiprazole and brexpiprazole.


Examine whether the Weber effect influences early post-marketing reporting patterns of brexpiprazole.


Develop a machine learning framework to predict individual patient risk of developing akathisia.



Chapter 2. Methodology
2.1 Data Source and Preprocessing
Data were obtained from the U.S. FDA open database (openFDA) Adverse Event Reporting System (FAERS).
Definition of Target Adverse Events: Target outcomes were defined using MedDRA terminology, including Akathisia, Extrapyramidal disorder, Dystonia, and Tremor.
Inclusion and Exclusion Criteria:
Only reports submitted by healthcare professionals were included.


Cases with age <10 years or >100 years were excluded as extreme outliers.


Duplicate cases were removed using report IDs and event dates, retaining only the most recent version.


After data cleaning, the dataset included drug names, active ingredients, and patient demographic characteristics (Figure 1).
Figure 1. Snapshot of FAERS data after preprocessing (Source: compiled from openFDA by this study).

2.2 Statistical Analysis: Disproportionality Analysis
A case/non-case approach was employed to calculate reporting odds ratios (RORs) and corresponding 95% confidence intervals, enabling detection of treatment-related adverse event signals.

2.3 Machine Learning Framework
Model Selection: XGBoost algorithm.
Validation Strategy:
Training set: Data from 2004–2024 with 5-fold cross-validation.


Test set: 2025 data reserved for temporal holdout validation.


Evaluation Metrics: Given the clinical priority of minimizing false negatives, the F-beta score (β = 3) was used as the primary metric, supplemented by micro-AUROC and micro-AUPR.

Chapter 3. Results
3.1 Drug Risk Comparison: Disproportionality Analysis
Analyses were first conducted on the full dataset (2004–2024), followed by a post-2019 subset (after stabilization of brexpiprazole use) to assess the Weber effect.
Figure 2. Forest plot comparing the risk of akathisia between aripiprazole and brexpiprazole (red dashed line indicates ROR = 1; values <1 with confidence intervals not crossing 1 indicate lower risk for aripiprazole).
Key findings from Figure 2 include:
All-time period: Aripiprazole exhibited an ROR of 0.60 (95% CI: 0.52–0.69), indicating a significantly lower risk than brexpiprazole.


Recent period (2019–2024): Even after accounting for early post-marketing reporting bias, aripiprazole remained associated with lower risk (ROR = 0.69; 95% CI: 0.58–0.81).



3.2 Machine Learning Model Performance
Overall model performance on the independent 2025 test set is summarized in Figure 3.
Figure 3. Machine learning model performance on 2025 test data, including confusion matrix, ROC curve, PR curve, and feature importance.
3.2.1 Confusion Matrix Analysis
The model demonstrated high sensitivity:
Recall: 70.4%. Of 27 patients who developed akathisia, 19 were correctly identified.


False positives: 181 false alerts were generated, resulting in low precision (9.5%).



3.2.2 Overall Discriminative Ability
ROC Curve: Micro-AUROC of 0.7833, indicating good overall discrimination.


PR Curve: Micro-AUPR of 0.2940. Although modest in absolute terms, this performance substantially exceeded random guessing given the severe class imbalance (baseline ≈ 0.05).



3.2.3 Feature Importance
Feature importance analysis revealed that, in addition to drug type, other neuropsychiatric symptoms (e.g., psychomotor hyperactivity, anxiety) were key predictors of akathisia risk.

Chapter 4. Discussion
4.1 Pharmacological Expectations versus Real-World Evidence
Theoretically, brexpiprazole’s high affinity for 5-HT1A receptors should mitigate EPS risk. However, FAERS data from this study demonstrated a significantly higher risk of akathisia. This discrepancy suggests that clinical trials may suffer from selection bias and may not fully capture the complexity of real-world patient populations.

4.2 Verification of the Weber Effect
The Weber effect posits a surge in adverse event reporting shortly after drug approval. In this study, the ROR increased from 0.60 (all-time) to 0.69 (post-2019), indicating partial influence of the Weber effect. Nevertheless, the ROR remained significantly below 1, confirming that the elevated risk associated with brexpiprazole cannot be attributed solely to reporting bias.

4.3 Clinical Positioning of the Model: High-Sensitivity Screening
The modeling strategy prioritized sensitivity over precision.
Strength: A recall of 70.4% makes the model suitable for preliminary screening, effectively capturing most high-risk cases.


Limitation: The low precision (9.5%) implies a high false-positive rate, which may contribute to alert fatigue. Therefore, model outputs should be interpreted as risk alerts rather than diagnostic conclusions.



4.4 Study Limitations
Class imbalance: The prevalence of akathisia in FAERS data was approximately 5%, limiting achievable AUPR.


Concept drift: Declines in test-set performance (AUPR 0.40 → 0.29) suggest temporal changes in patient characteristics or prescribing patterns.



Chapter 5. Conclusion
This study demonstrates that, in real-world data, brexpiprazole is associated with a higher risk of akathisia than aripiprazole, and that this finding cannot be fully explained by the Weber effect. Although the proposed machine learning model has a high false-positive rate, its strong sensitivity makes it a valuable adjunct for identifying high-risk patients. Clinicians should remain vigilant when prescribing brexpiprazole.

References
Miller, J. N., & Black, D. W. (2020). Bipolar disorder and suicide: A review. Current Psychiatry Reports, 22(2), 6.
Nierenberg, A. A., et al. (2023). Diagnosis and treatment of bipolar disorder: A review. JAMA, 330(14), 1370–1380.
Tarzian, M., et al. (2023). Cariprazine for treating schizophrenia, mania, bipolar depression, and unipolar depression: A review of its efficacy. Cureus, 15(5), e39309.
Yan, D., et al. (2025). Plasma proteome-driven liquid biopsy for individualized monitoring and risk stratification of immune-related adverse events in checkpoint immunotherapy. Molecular & Cellular Proteomics, 101488.
U.S. Food and Drug Administration (openFDA). FAERS database statistics and data documentation.


