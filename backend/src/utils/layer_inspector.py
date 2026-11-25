"""
layer_inspector.py -- Multi-Stage Intermediate Feature Extraction & Visualization Engine
Lightweight Topology-Aware MobileViT Network for Rural Road Extraction
"""

import os
import time
import numpy as np
import cv2
import onnx
import onnxruntime as ort

STAGE_METADATA = [
    {
        "id": "stage_0",
        "title": "Stage 0: Input Optical Satellite Raster",
        "node": "input_image",
        "type": "Input Raster",
        "shape_ratio": (1, 1),
        "channels": 3,
        "description": "Raw optical satellite tile (0.5m GSD). Shows rural unpaved tracks, foliage, tree canopy shadows, and varying soil contrast.",
        "math": r"X \in \mathbb{R}^{3 \times H \times W}, \quad \tilde{X} = \frac{X / 255 - \mu}{\sigma}",
        "insight": "High spatial resolution (0.5m GSD) reveals narrow dirt paths (2-6 px wide) that standard low-res models blur into background terrain."
    },
    {
        "id": "stage_1",
        "title": "Stage 1: Stem - Directional Strip Convolutions (1x3 -> 3x1)",
        "node": "/m/stem/act_1/Mul_output_0",
        "type": "Strip Convolution",
        "shape_ratio": (1/2, 1/2),
        "channels": 32,
        "description": "Factorized directional convolutions (1x3 horizontal then 3x1 vertical with stride 2). Filters out isotropic jungle noise and acts as an elongated road scanner.",
        "math": r"F_{\text{stem}}(X) = \text{SiLU}\left(\text{BN}_v(\text{Conv}_{3\times 1}(\text{SiLU}(\text{BN}_h(\text{Conv}_{1\times 3}(X)))))\right)",
        "insight": "Roads are tubular and elongated. Strip convolutions cut parameters by 33% (6 weights instead of 9) while boosting sensitivity to linear road corridors."
    },
    {
        "id": "stage_2",
        "title": "Stage 2: Encoder 1 - Inverted Residual Block",
        "node": "/m/enc1/conv/conv.6/Conv_output_0",
        "type": "Inverted Residual",
        "shape_ratio": (1/4, 1/4),
        "channels": 64,
        "description": "Depthwise separable convolution with inverted bottleneck expansion. Downsamples to 1/4 resolution, extracting localized textural contrast.",
        "math": r"F_{\text{enc1}} = \text{Conv}_{1\times 1}\left(\text{DWConv}_{3\times 3}\left(\text{Conv}_{1\times 1}(X_{\text{stem}})\right)\right) + X_{\text{res}}",
        "insight": "Distinguishes road surface roughness, dirt compaction, and tire tracks from surrounding vegetation and agricultural soil."
    },
    {
        "id": "stage_3",
        "title": "Stage 3: Encoder 2 - MobileViT v2 Linear Attention & Channel Shift",
        "node": "/m/enc2_mvit/fusion/fusion.2/Mul_output_0",
        "type": "Linear Transformer",
        "shape_ratio": (1/4, 1/4),
        "channels": 64,
        "description": "Channel Shift (+/-2px in 4 cardinal directions) followed by separable linear self-attention O(N*d). Starts linking separated road segments.",
        "math": r"\text{Attn}(Q, K, V) = \frac{\sum_{j} \text{Softmax}(K)_j V_j}{\sum_{j} \text{Softmax}(K)_j} \cdot Q, \quad \mathcal{O}(N \cdot d)",
        "insight": "Standard ViTs require O(N²) quadratic memory, which explodes on satellite tiles. Separable self-attention runs linearly O(N), linking road segments across tree shadows."
    },
    {
        "id": "stage_4",
        "title": "Stage 4: Encoder 3 - Deep Semantic MobileViT v2",
        "node": "/m/enc3_mvit/fusion/fusion.2/Mul_output_0",
        "type": "Linear Transformer",
        "shape_ratio": (1/8, 1/8),
        "channels": 96,
        "description": "Deep transformer block operating at 1/8 resolution. Suppresses non-road linear artifacts (crop borders, irrigation ditches, forest edges).",
        "math": r"Z_{\text{shift}} = \text{Concat}\left(Z_0[\uparrow 2], Z_1[\downarrow 2], Z_2[\leftarrow 2], Z_3[\rightarrow 2], Z_{\text{rest}}\right)",
        "insight": "Channel Shift expands the receptive field by +/-8px in original space at ZERO extra FLOPs or weights, allowing the model to see road continuity across tree patches."
    },
    {
        "id": "stage_5",
        "title": "Stage 5: Bottleneck - Deepest 3-Block Transformer Core",
        "node": "/m/bottleneck/fusion/fusion.2/Mul_output_0",
        "type": "Bottleneck Core",
        "shape_ratio": (1/16, 1/16),
        "channels": 128,
        "description": "Deepest latent space (128 channels, 1/16 resolution). Contains whole-scene topological structure and bridges massive forest canopy gaps.",
        "math": r"H_{\text{bottleneck}} = \mathcal{T}_3\left(\mathcal{T}_2\left(\mathcal{T}_1(Z_{\text{enc3}})\right)\right), \quad C=128",
        "insight": "Every pixel at this stage sees the entire scene. Global perspective is key to inferring roads hidden under continuous tree canopies."
    },
    {
        "id": "stage_6",
        "title": "Stage 6: Attention Gate 3 - Deep Skip Connection Filter",
        "node": "/m/gate3/psi/psi.2/Sigmoid_output_0",
        "type": "Attention Gate",
        "shape_ratio": (1/8, 1/8),
        "channels": 1,
        "description": "Soft spatial attention mask alpha in [0, 1]. Gating signal from the bottleneck dynamically suppresses background jungle textures from Encoder 3 skip features.",
        "math": r"\alpha_3 = \sigma\left(\psi^T \cdot \text{SiLU}(W_g \cdot g_{\text{up}} + W_x \cdot x_{\text{skip}})\right)",
        "insight": "Without attention gates, dense skip connections flood the decoder with misleading foliage edges. The gate acts as a spotlight, passing ONLY genuine road features."
    },
    {
        "id": "stage_7",
        "title": "Stage 7: Decoder 3 - First Spatial Upsample & Strip Conv",
        "node": "/m/dec3/act_1/Mul_output_0",
        "type": "Decoder Block",
        "shape_ratio": (1/8, 1/8),
        "channels": 96,
        "description": "Bilinear 2x upsampling concatenated with gated skip features, followed by factorized strip convolutions to begin spatial reconstruction.",
        "math": r"D_3 = \text{StripConv}\left(\text{Concat}\left(\text{Up}_2(H_{\text{bottleneck}}), \alpha_3 \odot x_{\text{enc3}}\right)\right)",
        "insight": "Combines coarse global semantic certainty with localized skip boundaries, initiating the sharp delineation of the road corridor."
    },
    {
        "id": "stage_8",
        "title": "Stage 8: Attention Gate 2 - Mid-Level Spatial Gate",
        "node": "/m/gate2/psi/psi.2/Sigmoid_output_0",
        "type": "Attention Gate",
        "shape_ratio": (1/4, 1/4),
        "channels": 1,
        "description": "Mid-level spatial gating coefficients focusing attention on road junctions, forks, and intersections while suppressing building rooftops and crop plots.",
        "math": r"\alpha_2 = \sigma\left(\psi^T \cdot \text{SiLU}(W_g \cdot D_3 + W_x \cdot x_{\text{enc2}})\right)",
        "insight": "Notice the bright highlights around complex junctions and road curves. The gate eliminates ambiguous field textures."
    },
    {
        "id": "stage_9",
        "title": "Stage 9: Decoder 2 - Mid-Level Spatial Recovery",
        "node": "/m/dec2/act_1/Mul_output_0",
        "type": "Decoder Block",
        "shape_ratio": (1/4, 1/4),
        "channels": 64,
        "description": "Upsampling to 1/4 resolution, restoring spatial continuity of narrow unpaved tracks and forest clearing paths.",
        "math": r"D_2 = \text{StripConv}\left(\text{Concat}\left(\text{Up}_2(D_3), \alpha_2 \odot x_{\text{enc2}}\right)\right)",
        "insight": "The road network emerges with high clarity. Narrow corridors are reconnected across shadow gaps."
    },
    {
        "id": "stage_10",
        "title": "Stage 10: Attention Gate 1 - High-Resolution Skip Filter",
        "node": "/m/gate1/psi/psi.2/Sigmoid_output_0",
        "type": "Attention Gate",
        "shape_ratio": (1/2, 1/2),
        "channels": 1,
        "description": "Fine-grained spatial attention gate operating at 1/2 resolution. Pins down precise road boundaries and edge pixels.",
        "math": r"\alpha_1 = \sigma\left(\psi^T \cdot \text{SiLU}(W_g \cdot D_2 + W_x \cdot x_{\text{stem}})\right)",
        "insight": "Filters out fine high-frequency noise (e.g. tree leaves, fence lines, tire ruts) from the stem skip connection."
    },
    {
        "id": "stage_11",
        "title": "Stage 11: Decoder 1 - Final High-Resolution Synthesis",
        "node": "/m/dec1/act_1/Mul_output_0",
        "type": "Decoder Block",
        "shape_ratio": (1/2, 1/2),
        "channels": 32,
        "description": "Final synthesis block producing a sharp, continuous tubular road feature map ready for binary classification.",
        "math": r"D_1 = \text{StripConv}\left(\text{Concat}\left(\text{Up}_2(D_2), \alpha_1 \odot x_{\text{stem}}\right)\right)",
        "insight": "Synthesizes fine-grain edge localization with deep semantic confidence. Road edges are crisp and uninterrupted."
    },
    {
        "id": "stage_12",
        "title": "Stage 12: Prediction Head - Continuous Road Probability Surface",
        "node": "road_probability",
        "type": "Sigmoid Head",
        "shape_ratio": (1, 1),
        "channels": 1,
        "description": "Final 1x1 projection followed by Sigmoid activation. Outputs a continuous confidence surface P(road) in [0, 1] for every pixel.",
        "math": r"P(x, y) = \sigma\left(\text{Conv}_{1\times 1}(\text{Up}_2(D_1))\right) \in [0, 1]",
        "insight": "Trained with composite BCE + Dice + clDice loss. The soft surface retains high confidence even under partial canopy cover."
    }
]


class MultiStageInspector:
    """Manages multi-output ONNX execution and feature map visualization."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.output_names = []
        self._build_multi_output_session()

    def _build_multi_output_session(self):
        """Construct an ONNX session that exposes all intermediate layer outputs."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        model = onnx.load(self.model_path)
        added_names = set(o.name for o in model.graph.output)

        for meta in STAGE_METADATA:
            node_name = meta["node"]
            if node_name != "input_image" and node_name not in added_names:
                vi = onnx.helper.ValueInfoProto()
                vi.name = node_name
                model.graph.output.append(vi)
                added_names.add(node_name)

        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False
        self.session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run_multi_stage_inference(self, tensor: np.ndarray):
        """Run single forward pass and return a dictionary mapping node name -> feature array."""
        start = time.perf_counter()
        raw_outputs = self.session.run(None, {"input_image": tensor})
        duration = time.perf_counter() - start

        res_dict = {}
        for name, arr in zip(self.output_names, raw_outputs):
            res_dict[name] = arr

        return res_dict, duration

    @staticmethod
    def render_feature_visualization(
        feat_arr: np.ndarray,
        mode: str = "Maximum Feature Energy (Peak Response)",
        colormap_id: int = cv2.COLORMAP_VIRIDIS,
        target_size: tuple = None,
        channel_idx: int = None
    ) -> np.ndarray:
        """
        Convert multi-channel feature tensor into a high-contrast, publication-grade heatmap.
        Uses percentile-based contrast stretching and smooth cubic interpolation so layer
        activations show distinct road corridors rather than washed-out noise.
        """
        if feat_arr.ndim == 4:
            feat_arr = feat_arr[0]

        if feat_arr.ndim == 3:
            C, H, W = feat_arr.shape
            if C == 1:
                # Attention Gate or Probability Map (values in [0, 1])
                act = feat_arr[0].astype(np.float32)
            elif channel_idx is not None and 0 <= channel_idx < C:
                # Specific individual channel requested
                act = feat_arr[channel_idx].astype(np.float32)
            elif "Maximum Feature Energy" in mode or "Peak" in mode:
                # Rectified peak activation across filters: reveals where ANY directional filter fired
                act = np.max(np.maximum(feat_arr, 0), axis=0).astype(np.float32)
            elif "L2" in mode or "Magnitude" in mode:
                # L2 norm across feature channels
                act = np.linalg.norm(np.maximum(feat_arr, 0), axis=0).astype(np.float32)
            elif "Top Active" in mode:
                # Channel with highest variance / dynamic range
                variances = np.var(feat_arr, axis=(1, 2))
                top_idx = int(np.argmax(variances))
                act = feat_arr[top_idx].astype(np.float32)
            else:
                # Positive activation mean (avoids negative SiLU cancellation)
                act = np.mean(np.maximum(feat_arr, 0), axis=0).astype(np.float32)
        elif feat_arr.ndim == 2:
            act = feat_arr.astype(np.float32)
        else:
            raise ValueError(f"Unexpected shape for feature map: {feat_arr.shape}")

        # Robust percentile normalization (1st to 99th percentile) for rich dynamic contrast
        if act.size > 0:
            p_low = float(np.percentile(act, 1))
            p_high = float(np.percentile(act, 99))
            if p_high > p_low:
                normed = np.clip((act - p_low) / (p_high - p_low), 0.0, 1.0)
            else:
                normed = np.zeros_like(act, dtype=np.float32)
        else:
            normed = act

        norm_u8 = (normed * 255.0).astype(np.uint8)

        # Smooth cubic interpolation to target size (prevents blocky pixelated artifacts on deep layers)
        if target_size is not None and (norm_u8.shape[1], norm_u8.shape[0]) != target_size:
            norm_u8 = cv2.resize(norm_u8, target_size, interpolation=cv2.INTER_CUBIC)

        colored_bgr = cv2.applyColorMap(norm_u8, colormap_id)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        return colored_rgb

    @staticmethod
    def render_channel_grid(
        feat_arr: np.ndarray,
        num_channels: int = 8,
        colormap_id: int = cv2.COLORMAP_VIRIDIS,
        cell_size: int = 128
    ) -> list:
        """
        Render a list of individual channels with their channel index and formatted images.
        """
        if feat_arr.ndim == 4:
            feat_arr = feat_arr[0]
        if feat_arr.ndim != 3:
            return []

        C, H, W = feat_arr.shape
        num_to_show = min(num_channels, C)

        # Pick channels with highest variance (most informative directional features)
        variances = np.var(feat_arr, axis=(1, 2))
        top_indices = np.argsort(variances)[::-1][:num_to_show]

        results = []
        for ch_idx in top_indices:
            ch_data = feat_arr[ch_idx]
            p_low, p_high = float(np.percentile(ch_data, 2)), float(np.percentile(ch_data, 98))
            if p_high > p_low:
                normed = np.clip((ch_data - p_low) / (p_high - p_low), 0.0, 1.0)
            else:
                normed = np.zeros_like(ch_data, dtype=np.float32)
            u8 = (normed * 255.0).astype(np.uint8)
            u8_resized = cv2.resize(u8, (cell_size, cell_size), interpolation=cv2.INTER_CUBIC)
            bgr = cv2.applyColorMap(u8_resized, colormap_id)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results.append((int(ch_idx), rgb, float(ch_data.min()), float(ch_data.max())))

        return results

    @staticmethod
    def create_blended_overlay(
        base_rgb: np.ndarray,
        heatmap_rgb: np.ndarray,
        alpha: float = 0.55
    ) -> np.ndarray:
        """Blend optical satellite raster with feature heatmap."""
        h, w = base_rgb.shape[:2]
        if heatmap_rgb.shape[:2] != (h, w):
            heatmap_resized = cv2.resize(heatmap_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            heatmap_resized = heatmap_rgb

        blended = cv2.addWeighted(base_rgb, 1.0 - alpha, heatmap_resized, alpha, 0)
        return blended
