**Introduction**

- Background: large-scale perturbation transcriptomics (LINCS L1000); predicting chemical-gene interactions (CGI) from structure; challenge of generalization to unseen compounds (chemical cold split)

- Current Limitations: existing methods (DeepCE, CIGER, PRnet) treat drug-gene interaction as scalar regression or classification without mechanistic structure; lack intrinsic interpretability; rely on task-mismatched objectives or pretraining

- Our Contributions:
  1. Spectral operator decomposition: drug-gene interaction modeled as T = I + UΣVᵀ (low-rank perturbation), yielding mode-specific amplitude σⱼ, direction Uⱼ, and gene response Vⱼ
  2. End-to-end interpretable: no post-hoc attribution; spectral modes emerge from training objective
  3. SpectrumDirectionCL: contrastive loss enforcing distinct directional fingerprints across drugs
  4. Biologically validated: 8 spectral modes correspond to 8 GO biological processes (7/8 FDR<0.05); drug σ fingerprints match known MOA

---

**Results**

**1. Performance Comparison**

- 1.1 Benchmark Comparison: 4 cell lines (MCF7, A375, A549, VCAP), 5-fold chemical cold CV; AUC main metric; comparison to DeepCE-CLS (fairest baseline, same task formulation)

- 1.2 Ablation: operator structure contributes most (−0.0025 without); CL reduces variance; ortho regularization ensures mode independence; GIN vs fixed ECFP4 (+0.0236)

**2. Biological Interpretability of Spectral Modes**

- 2.1 GO Enrichment of Spectral Modes: 8 modes → 8 distinct GO biological processes via interaction-view gene ranking; Mode 5 (NF-κB/MAPK) most significant (FDR=2.1e-3); 7/8 modes below FDR=0.05 threshold

- 2.2 Drug Mechanism-of-Action Fingerprinting: per-drug σ z-score profiles reveal MOA signatures; 7 representative drugs all show z>2 in expected mode; trametinib (MEK inhibitor) Mode 4 z=+5.7, Mode 1 z=+5.0; fenretinide (apoptosis stimulant) activates M1+M4+M7 simultaneously

- 2.3 Drug Clustering by Spectral Signature: UMAP of per-drug σ profiles reveals MOA-driven clustering; active drugs separate from inactive by dominant mode

**3. Cross-Cell Line Generalization**

- 3.1 Performance across 4 cell lines: MCF7 0.8928±0.0021, A375 0.9011±0.0050, A549 TBD, VCAP TBD

- 3.2 (Optional) Cross-cell mode consistency: do the same 8 biological processes emerge independently in A375/A549/VCAP? (requires running interp pipeline on other models)

---

**Discussion**

- Key Findings: spectral decomposition provides intrinsic interpretable structure while maintaining competitive AUC; mode identity is data-driven and validated post-hoc

- Methodological Advantages: end-to-end without pretrained embeddings; operator rank r=8 as inductive bias for pathway-level organization

- Biological Insights: NF-κB and Kinase signaling are the dominant drug-response axes in MCF7 (Mode 0, 5 most significant GO); apoptosis-inducing drugs activate multiple modes simultaneously (fenretinide: M1+M4+M7)

- Limitations: chemical cold split is stringent; mode labels are post-hoc (from GO); Mode 6 (RTK) marginal significance; 4 cell lines only

- Future Directions: 46-cell line generalization; cross-cell mode alignment; integration with clinical response data

---

**Methods**

**1. Data**

- 1.1 Dataset: LINCS L1000 (978 landmark genes); 4 cell lines; binary label: |z-score| > 2.0 = positive pair; chemical cold 5-fold split by compound Tanimoto similarity

- 1.2 Statistics: MCF7 209,657 pairs / 11,933 compounds; positive rate ≈ 7%; val fold ~42K pairs, ~9.7K compounds

**2. Model Architecture (DrugOperatorNet)**

- 2.1 Chemical Encoder: 3-layer GIN (hidden=128, edge features: bond type/ring/aromatic); global mean/max pool; output → pharma_emb [r, H] via linear projection

- 2.2 Gene Encoder (GeneMultiHeadReader): embedding table (978 genes × H); r independent attention heads reading per-mode gene representation; output h_g_modes [r, H], h_g_global [H]

- 2.3 Perturbation Operator: T = I + UΣVᵀ where U=pharma_emb [r,H], Σ=diag(σ) per-mode amplitudes, V=h_g_modes [r,H]; interaction spectrum = ⟨U_j, V_j⟩·σ_j; perturbation Δh = Σⱼ spectrum_j · V_j

- 2.4 Classifier: MLP on [h_g_global; Δh] → logit; sigmoid → probability

- 2.5 SpectrumDirectionCL: contrastive loss on direction vectors U_j normalized to unit sphere; push apart directions of different drugs; τ=0.07

**3. Training**

- 3.1 Objective: BCE + lam_sparse·||σ||₁ + lam_ortho·||UᵀU − I||_F² + lam_cl·L_CL

- 3.2 Optimizer: Adam; lr=2e-4 with 5-epoch warmup; cosine decay with restart; patience=10 early stopping; batch=512; AMP

- 3.3 Hyperparameters: r=8, H=128, lam_sparse=0.01, lam_ortho=0.1, lam_cl=0.1, dropout=0.3, seed=42

**4. Interpretability Pipeline**

- 4.1 GO Enrichment: for each mode j, rank genes by interaction view (mean spectrum_j over positive pairs); top-100 genes → Enrichr (GO_BP_2023, KEGG_2021); FDR-corrected

- 4.2 Drug Fingerprinting: per-drug mean σ across val pairs; z-score vs global distribution; dominant mode = argmax(z-score)

**5. Baselines**

- DeepCE-CLS: original DeepCE architecture with masked BCE classification head (same task formulation as ours); 100 epochs, MCF7 Fold0 only

- DeepCE-REG: original regression objective; AUC computed by thresholding predicted z-scores at 2.0

- ECFP4: fixed 2048-bit fingerprint + linear projection; same operator structure; ablation of GIN

