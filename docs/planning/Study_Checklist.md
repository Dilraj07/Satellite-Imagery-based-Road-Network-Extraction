# 📚 GeoScribe: Comprehensive Study Checklist

**Objective:** Prepare for building a patent-worthy Real-Time Geospatial & AI Infrastructure Tracking Platform.

## Phase 1: Deep Learning for Remote Sensing (AI Extraction)

_The core engine that extracts roads from raw satellite imagery._

- [ ] **CNN Foundations & Semantic Segmentation**
  - Read: "U-Net: Convolutional Networks for Biomedical Image Segmentation".
  - Understand: Encoder-decoder structures, skip connections, and Receptive Fields.
  - Implement: A basic U-Net in PyTorch on a toy dataset.
- [ ] **Advanced Architectures for Linear Features**
  - Study: D-LinkNet (Winner of DeepGlobe Road Extraction Challenge).
  - Understand: Dilated convolutions and how they preserve spatial resolution without losing context.
- [ ] **Vision Transformers (ViTs)**
  - Read: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".
  - Understand: Self-attention mechanisms vs. Convolutions.
  - Study: SegFormer for efficient semantic segmentation.

## Phase 2: Topology and Graph Neural Networks

_Ensuring the extracted pixels form a highly connected, routable graph._

- [ ] **Graph Theory Basics & Graph Neural Networks (GNNs)**
  - Understand: Nodes (Intersections), Edges (Road Segments), and Adjacency Matrices.
  - Read: "A Comprehensive Survey on Graph Neural Networks".
- [ ] **Topology-Aware Extraction Models**
  - Study: _RoadTracer_ (Iterative tracing approach).
  - Study: _Sat2Graph_ (Direct-to-graph extraction without pixel bottlenecks).
- [ ] **Metrics for Functional Connectivity**
  - Learn: IoU (Intersection over Union) vs APLS (Average Path Length Similarity).
  - Understand why high pixel accuracy does not equal a good road network.

## Phase 3: Geospatial Engineering & 3D Web Rendering

_The visualization engine._

- [ ] **WebGIS & 3D Rendering Frameworks**
  - Learn: `CesiumJS` or `Deck.gl` combined with `React/Next.js`.
  - Goal: Render a 3D Earth globe in the browser.
- [ ] **Google Photorealistic 3D Tiles**
  - Read: Google Maps Platform 3D Tiles documentation.
  - Understand: OGC 3D Tiles specification and Level of Detail (LOD) streaming.
- [ ] **Vector Data Formats**
  - Understand the difference between `GeoJSON`, `TopoJSON`, and Vector Tiles (`.mvt`).

## Phase 4: Live Data Ingestion & API Pipelines

_The dynamic layer._

- [ ] **Real-Time Data Architectures**
  - Learn: WebSockets vs. REST Polling vs. Server-Sent Events (SSE).
- [ ] **Orbital Mechanics (Space-Track API)**
  - Learn: What a TLE (Two-Line Element) is.
  - Study: The `satellite.js` library to calculate live lat/lon from a TLE.
- [ ] **Aviation Tracking (OpenSky / ADS-B Exchange)**
  - Learn: How ADS-B transponders work.
- [ ] **Seismic & IoT Data**
  - Explore: USGS Earthquake API format and Austin Open Data portal queries.

## 💡 Patentability Research (Simultaneous)

- [ ] **Novelty Search:** Investigate existing patents combining _Static AI Topological Generation_ combined with _Real-Time Dynamic Aviation/Orbital/Seismic Overlays_.
- [ ] **Identify the "Inventive Step":** Is the novelty in the _data fusion pipeline_, the _graph-rendering optimization_, or the specific _use-case_ (e.g., automated disaster relief routing based on live seismic events overlaid on AI-extracted roads)?
