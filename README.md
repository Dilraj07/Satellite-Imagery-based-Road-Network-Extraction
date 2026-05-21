# COE Project: Auditing Rural Connectivity

## MobileViT + clDice for Rural Road Topological Extraction

In developing nations, the infrastructure audit and mapping required under initiatives such as the PMGSY drive an urgent need to accurately map millions of kilometers of rural, unpaved roads. Current manual auditing workflows are prohibitively labor-intensive and slow. While standard deep learning semantic segmentation models have excelled at urban highway mapping, they exhibit severe performance drop-offs (Domain Shift) when deployed in rural environments.

### Core Project Objective
Construct an ultra-lightweight encoder-decoder architecture utilizing **MobileViT v2** as a fast, low-parameter backbone, synthesized with graph-theoretic/topology-preserving constraints (**clDice Loss**) to enforce contiguous, navigable road predictions on Indian rural topographies while optimizing for edge-device deployment.

---

## 4-Member Project Structure

- **Member 1 (Lead Architect):** Designing and building the core lightweight encoder-decoder architecture using MobileViT v2. 
- **Member 2 (Graph & Loss Engineer):** Implementing the graph-theoretic clDice loss function and morphological skeletonization layers.
- **Member 3 (Data & Pipeline Engineer):** Preprocessing the DeepGlobe Dataset, implementing weak supervision layers aligning OSM centerlines.
- **Member 4 (Validation & MLOps):** Building validation suites specializing in non-pixel routing metrics (APLS).

---

## Technical Approach
1. **Backbone Encoder:** MobileViT v2 replaces heavy Vision Transformers and CNNs, lowering parameter footprints.
2. **Loss Function:** Replaces standard IoU/BCE with graph-theoretic **clDice Loss** to calculate overlap explicitly on the topological graph.
3. **Weak Supervision:** Uses 1-pixel-wide vector line-strings extracted from OpenStreetMap (OSM) to eliminate manual annotations.

---

## Phase 1 (Week 1)
- **Member 1 Focus:** Initialize repository framework, PyTorch environment, and baseline U-Net architectures.
- *(Ongoing initialization in this phase...)*
