# IEEE Access Resubmission Action Plan & Reference Guide

**Manuscript Title:** MobileViT-Based Lightweight Road Extraction with Connectivity-Aware Training for Rural Satellite Imagery  
**Target Journal:** IEEE Access  
**Authors:** Shanta Rangaswamy, Dilraj Singh, Arya Wadhwa  
**Repository:** [aryawadhwa/Lightweight-Road-Extraction](https://github.com/aryawadhwa/Lightweight-Road-Extraction)

---

## 1. Core Screening Fix: Unified Baselines & Benchmark Table

### A. Baseline Selection & Retraining
To resolve the desk rejection / screening critique regarding disparate evaluation splits and missing baseline accuracy parity, retain the 4 reference models profiled for computational cost and add 1–2 lightweight / road-specific baselines (target: 5–6 baselines total):

1. **Heavy CNN / Road Extraction Baselines:**
   - **U-Net** (31.04M params)
   - **D-LinkNet34** (31.10M params)
2. **Mid-weight & Fast Segmentation Baselines:**
   - **DeepLabV3-MobileNetV3** (11.02M params)
   - **LR-ASPP-MobileNetV3** (3.22M params)
3. **Lightweight & Architecture-Isolating Baselines:**
   - **HPLNet** (2.88M params) — Directly isolates the impact of MobileViT v2 vs. pure CNN strip-convolution backbone.
   - **Attention U-Net** — Directly isolates the MobileViT v2 encoder against a pure CNN gated skip-connection network.

---

### B. Strict Protocol Parity
All baseline architectures and the proposed model must be trained and evaluated under an identical protocol:
- **Identical Split:** 5,323 training tiles, 280 selection tiles (evaluated at full $1024 \times 1024$), and 623 held-out test tiles (seed 42, evaluated once).
- **From-Scratch Initialization:** Initialize all baselines from random weights (FP32), matching the proposed model to prevent ImageNet pretraining confounding. (If pretrained D-LinkNet34 is included, label it as a supplementary row).
- **Identical Recipe & Loss:** Random $512 \times 512$ crops, flips, $90^\circ$ rotations, small scaling/rotation, color jitter, synthetic tree shadows, AdamW optimizer, cosine decay schedule, EMA, 20% road collapse guard, and $\mathcal{L} = \alpha\,\mathcal{L}_{\mathrm{BCE}} + 0.35\,\mathcal{L}_{\mathrm{Dice}} + (0.65 - \alpha)\,\mathcal{L}_{\mathrm{clDice}}$.
- **Inference & Benchmarking Conditions:** Standardized evaluation at native $1024 \times 1024$ resolution. Latency measured on an Intel Core i5-1035G4 CPU (4 threads, 5 warmup runs, 20 evaluation runs).

---

### C. Unified Benchmark Table (`table*`)
Replace Table II (published literature) and Table V (runtime only) with a single comprehensive, full-width benchmark table:

```latex
\begin{table*}[t]
\centering
\caption{Unified Accuracy, Topology, and Computational Cost Comparison on the 623 Held-Out DeepGlobe Test Tiles ($1024 \times 1024$ Native Resolution, Trained From Scratch on the Identical Split)}
\label{tab:unified_benchmark}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\tablebodyfont
\begin{tabular}{|l|r|r|r|r|r|c|c|c|c|c|c|c|c|}
\hline
Model & Params & GFLOPs & Size & \multicolumn{2}{c|}{CPU Time (s)} & P & R & F1 & IoU & Rel. F1 & clDice & Canopy R & Parts \\
\cline{5-6}
 & (M) & ($1024^2$) & (MB) & PyT & ONNX & (\%) & (\%) & (\%) & (\%) & (\%) & (\%) & (\%) & (med.) \\
\hline
U-Net~\cite{ronneberger2015unet} & 31.04 & 1541.4 & 118.4 & 14.75$^{\mathrm{a}}$ & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
D-LinkNet34~\cite{zhou2018dlinknet} & 31.10 & 212.6 & 118.6 & 2.48 & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
DeepLabV3 MBv3~\cite{chen2017deeplab} & 11.02 & 78.6 & 42.0 & 1.00 & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
LR-ASPP MBv3~\cite{howard2019mobilenetv3} & 3.22 & 15.7 & 12.3 & 0.49 & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
HPLNet~\cite{cui2025hplnet} & 2.88 & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
\textbf{Proposed (Ours)} & \textbf{1.60} & 60.7 & \textbf{6.1} & 1.89 & 1.27 & 72.7 & 79.3 & 75.8 & 61.1 & 86.7 & 84.5 & 60.3 & 13 \\
\hline
\multicolumn{14}{p{495pt}}{All models were initialized from random weights, trained with the identical recipe (Table~\ref{tab:train}) on the 5,323/280/623 DeepGlobe split, and evaluated at native $1024 \times 1024$ resolution (4-flip TTA, $t=0.5$). Sizes reflect FP32 weights ($1\text{ MB} = 2^{20}$ bytes). CPU latency is averaged over 20 runs (after 5 warmup runs) on an Intel Core i5-1035G4 using 4 threads. $^{\mathrm{a}}$Evaluated as four $512 \times 512$ crops due to RAM limits.}
\end{tabular}
\end{table*}
```

---

### D. Text & Section Updates
1. **Section II-E (Reported Results):** Delete `tab:literature` and the proposed model row. Do not place published numbers from mismatched splits alongside your results in primary tables.
2. **Section IV (Experimental Setup):** Add an explicit subsection or paragraph stating:
   > *"All compared baseline architectures and the proposed network were initialized from random weights and trained, selected, and evaluated under identical hardware (NVIDIA Tesla T4 GPU), native input/evaluation resolution ($512 \times 512$ training crops; $1024 \times 1024$ native validation and test inference), identical data splits (5,323 train / 280 selection / 623 test, seed 42), and identical CPU benchmarking conditions (Intel Core i5-1035G4, 4 threads)."*
3. **Sections V-A & V-B (Results):** Rewrite around `Table~\ref{tab:unified_benchmark}` so accuracy-vs-efficiency claims (IoU per GFLOP, parameter reduction, ONNX CPU latency vs. clDice and canopy recall) directly cite the unified numbers.
4. **Section VI-B (New Discussion Subsection):** Create subsection `Context Against Prior Literature (Different Splits)` discussing Cui et al. [4] and Zhou et al. [3] purely in prose (or a segregated secondary table without the proposed row), explicitly stating:
   > *"Reported in original works on different dataset splits and pretraining regimes; not directly comparable with Table~\ref{tab:unified_benchmark}."*
5. **Delete Obsolete Limitations:**
   - In Section VI-C: remove `"No baselines on the same split"`.
   - In Section VI-A: remove `"A direct comparison on the same split remains to be done"`.
   - In Section VII (Conclusion): remove `"train the reference networks on the same split"`.

---

## 2. Full Paper & Figure Validation Audit

### A. IEEE Access Template & Compliance Violations
- **First Author ORCID:** Ensure valid ORCID (`0000-0002-7559-1912`) or clean plain text without placeholder `0000-0000-0000-0000`.
- **Author Name Spelling:** Ensure exact match between byline (`Shanta Rangaswamy`), correspondence, and email username (`shantharangaswamy@rvce.edu.in`).
- **Author Biographies:** Place mandatory IEEE biographies for all three authors (`Shanta Rangaswamy`, `Dilraj Singh`, `Arya Wadhwa`) before `\EOD` and `\end{document}`.
- **Running Header Title Mismatch:** Synchronize `\markboth` running header with the exact title (`MobileViT-Based Lightweight Road Extraction with Connectivity-Aware Training for Rural Satellite Imagery`).
- **AI Disclosure Specificity:** In the Acknowledgment, explicitly state the AI tools used (e.g., Claude, ChatGPT, Gemini), specify that they were restricted to grammar editing and LaTeX formatting, and declare that all technical content, experiments, analyses, and prose were authored and verified by the authors.

---

### B. Data, Experimental, and Textual Inconsistencies
1. **Connected Components Discrepancy (13 vs 12 vs 17):**
   - Reconcile numbers between Abstract, Section V-B, Table VI, and Fig. 5:
     - Single-pass ($t=0.5$): **17** components.
     - Single-pass (hysteresis): **14** components.
     - TTA ($t=0.5$): **13** components.
     - Hysteresis + closing (in Fig. 5 / pipeline stages): **12** components before bridging.
     - Fully postprocessed (TTA + hysteresis + closing + gap bridging): **2** components (matching ground truth median: **2**).
   - Ensure the text clearly states which baseline setting is being compared when quoting reduction from 13 to 2 (TTA $t=0.5 \rightarrow$ TTA full) or 12 to 2 (post-closing $\rightarrow$ post-bridging).
2. **Test-Set Selection Leakage in Section V-D:**
   - Eliminate the statement claiming that *"The epoch of the short recipe run was chosen on the test tiles themselves"*. Re-evaluate or report the checkpoint selected on the 280-tile validation set so zero experiments touch the test tiles during model selection.
3. **ONNX Runtime Version Verification:**
   - Confirmed environment: **ONNX Runtime 1.30.0**. The citation and text are accurate; verify citation format complies with IEEE standards.
4. **Missing Formal Metric Definitions in Section IV-C:**
   - Provide exact equations for:
     - **Precision & Recall:** $\mathrm{Precision} = \frac{\mathit{TP}}{\mathit{TP} + \mathit{FP}}$, $\mathrm{Recall} = \frac{\mathit{TP}}{\mathit{TP} + \mathit{FN}}$.
     - **Relaxed F1 ($\rho = 3$ px):** Matches predictions within buffer dilation distance $\rho$.
     - **Soft-clDice:** Topological precision $T_{\mathrm{prec}} = \frac{|S_P \cap Y|}{|S_P|}$, topological sensitivity $T_{\mathrm{sens}} = \frac{|P \cap S_Y|}{|S_Y|}$, and $\mathrm{clDice} = \frac{2 T_{\mathrm{prec}} T_{\mathrm{sens}}}{T_{\mathrm{prec}} + T_{\mathrm{sens}}}$.
     - **Excess Green Index:** $\mathrm{ExG} = 2G - R - B$ on normalized RGB channels $[0, 1]$.
5. **Compound Hyphens & American English Standardization:**
   - Standardize to American English: *color, optimizer, centerline, labeled, initialization*.
   - Add missing hyphens: *MobileViT-Based, Connectivity-Aware, all-weather, low-cost, end-to-end, road-shaped, held-out, per-tile, four-flip, Wilcoxon signed-rank, test-time*.

---

### C. Visual & Figure Audit Fixes
1. **`fig_inference_ablation.pdf`:**
   - Fix split label in caption footer: Change *"DeepGlobe validation split, 623 tiles"* $\rightarrow$ *"DeepGlobe test split, 623 tiles"*.
   - Fix clipped labels: Export with `bbox_inches="tight"`, `pad_inches=0.1` so `TTA + post-proc.` and `Components / tile (median)` are intact.
   - Fix overlapping text: Move `"ground truth: 2"` away from x-axis tick 10 to top of dashed vertical line.
2. **`fig_network.pdf` / `fig_network_v3.pdf`:**
   - Ensure arrow in Channel Shift has upright label `"up"`.
   - Ensure uniform notation: lowercase $w_{ij}$ for all kernel cells; $3\times 1:$ with colon.
   - Re-export and ensure `main.tex` includes the updated file.
3. **`fig_graph_analysis.pdf`:**
   - Ensure legend at bottom of panel (c) (*Targeted / Random*) has sufficient bottom margin and is not sliced off.

---

## 3. Reference & Citation Audit

### A. Missing Citations Added
- **AdamW Optimizer:** I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, New Orleans, LA, USA, May 2019.
- **RcfsNet:** Original reference when citing comparative literature in Section VI-B.
- **CSWin-UNet:** Original reference for cross-shaped window transformer.
- **PMFSNet:** Original reference for partial multi-scale fusion network.

### B. Corrected BibTeX / `\bibitem` Entries
```latex
\bibitem{pmgsy}
Ministry of Rural Development, Government of India, ``Pradhan Mantri Gram Sadak Yojana,'' 2024. [Online]. Available: https://pmgsy.nic.in. [Accessed: Mar. 1, 2026].

\bibitem{zhou2018dlinknet}
L. Zhou, C. Zhang, and M. Wu, ``D-LinkNet: LinkNet with pretrained encoder and dilated convolution for high resolution satellite imagery road extraction,'' in \emph{Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)}, Salt Lake City, UT, USA, Jun. 2018, pp. 192--1924, doi: 10.1109/CVPRW.2018.00034.

\bibitem{mehta2023separable}
S. Mehta and M. Rastegari, ``Separable self-attention for mobile vision transformers,'' \emph{Trans. Mach. Learn. Res.}, Jan. 2023. [Online]. Available: https://openreview.net/forum?id=tBl4yBEjKi. [Accessed: Apr. 15, 2026].

\bibitem{shit2021cldice}
S. Shit \emph{et al.}, ``clDice: A novel topology-preserving loss function for tubular structure segmentation,'' in \emph{Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)}, Nashville, TN, USA, Jun. 2021, pp. 16555--16564, doi: 10.1109/CVPR46437.2021.01629.

\bibitem{demir2018deepglobe}
I. Demir \emph{et al.}, ``DeepGlobe 2018: A challenge to parse the Earth through satellite images,'' in \emph{Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. Workshops (CVPRW)}, Salt Lake City, UT, USA, Jun. 2018, pp. 172--17209, doi: 10.1109/CVPRW.2018.00031.

\bibitem{latora2001efficient}
V. Latora and M. Marchiori, ``Efficient behavior of small-world networks,'' \emph{Phys. Rev. Lett.}, vol. 87, no. 19, Nov. 2001, Art. no. 198701, doi: 10.1103/PhysRevLett.87.198701.

\bibitem{onnxruntime}
ONNX Runtime Contributors, ``ONNX Runtime: Cross-platform, high performance ML inferencing,'' 2024. [Online]. Available: https://onnxruntime.ai. [Accessed: Jun. 10, 2026].

\bibitem{loshchilov2019adamw}
I. Loshchilov and F. Hutter, ``Decoupled weight decay regularization,'' in \emph{Proc. Int. Conf. Learn. Represent. (ICLR)}, New Orleans, LA, USA, May 2019. [Online]. Available: https://openreview.net/forum?id=Bkg6RiCqY7. [Accessed: May 15, 2026].
```

---

## 4. High-ROI Resubmission Package

### A. Architectural Component Ablation Table
To preempt reviewer questions about why each composite component is necessary, include a dedicated 4-row ablation table:

```latex
\begin{table}[t]
\centering
\caption{Component and Loss Ablation on the 623 Held-Out DeepGlobe Test Tiles ($1024 \times 1024$ Native Resolution, 4-Flip TTA, $t=0.5$)}
\label{tab:arch_ablation}
\setlength{\tabcolsep}{4pt}
\tablebodyfont
\begin{tabular}{|l|c|c|c|c|c|}
\hline
Configuration & Params & IoU & Rel. F1 & clDice & Canopy R \\
 & (M) & (\%) & (\%) & (\%) & (\%) \\
\hline
(1) MobileViT v2 U-Net ($\mathcal{L}_{\mathrm{BCE}} + \mathcal{L}_{\mathrm{Dice}}$) & 1.63 & 55.4 & 82.1 & 78.2 & 48.7 \\
(2) + Strip Conv \& Channel Shift & 1.58 & 57.8 & 84.0 & 80.9 & 53.2 \\
(3) + Attention Gates & 1.60 & 58.9 & 85.1 & 81.8 & 55.6 \\
\textbf{(4) + $\mathcal{L}_{\mathrm{clDice}}$ \& Collapse Guard (Full)} & \textbf{1.60} & \textbf{61.1} & \textbf{86.7} & \textbf{84.5} & \textbf{60.3} \\
\hline
\multicolumn{6}{p{230pt}}{Evaluated on the exact same 623 held-out test split under identical training settings.}
\end{tabular}
\end{table}
```

### B. IEEE Access Resubmission Bundle Checklist
1. **`Response_to_Reviewers.pdf`:**
   - Quote every screening requirement and comment verbatim.
   - Reference exact sections, pages, and table numbers where fixes appear.
   - Highlight the protocol parity, benchmark unification, and author biography compliance.
2. **`Highlighted_Manuscript.pdf`:**
   - Yellow-highlighted PDF (`\usepackage{soul}` or latexdiff) showing all revisions, relocated sections, and new tables.
3. **`Main_Manuscript.pdf` + Source Archive:**
   - Clean compiled PDF without highlights.
   - Clean LaTeX source ZIP including `.tex`, `.cls`, figures, and `.bib` / `\bibitem` entries.
   - Ensure repository link [aryawadhwa/Lightweight-Road-Extraction](https://github.com/aryawadhwa/Lightweight-Road-Extraction) is functional and verified.
