# MEMORY BOX: AI-Based Road Extraction & Live Geospatial Tracking

**Last Updated:** February 26, 2026

## 🧠 What is this file?

This file serves as the core persistent memory for the AI assistant across sessions. **Whenever you restart work on this project, reading this file is your absolute first priority.**

## 🎯 Project Vision

## 🎯 Project Vision (Patent & SOTA Paper Focused)

To invent and build a comprehensive **Real-Time Spatial Computing Engine** that dynamically fuses **Static AI-extracted infrastructure (Topology-Aware Road Networks)** with **Dynamic Live Tracking data (Satellites, Flights, Earthquakes, CCTV)**. The patentable novelty resides in generating a real-time interaction matrix between autonomous, deep-learning-generated ground topologies and multi-altitude, real-time spatial events.

## 📄 Core Context & Inputs

1. **Research Plan (Google Gemini.pdf):**
   - **Domain 1:** Pixel-Level Segmentation Architectures (U-Net, ResUNet, D-LinkNet, Transformers).
   - **Domain 2:** Topology-Aware and Graph-Based Models (GNNs, RoadTracer, Sat2Graph).
   - **Domain 3:** Datasets & Evaluation (SpaceNet, DeepGlobe, APLS metrics, Occlusion handling).
   - _Goal of Research:_ Accurately identify and extract road networks from high-res satellite imagery to form the static baseline topology.

2. **Geospatial & 3D Mapping Backbone:**
   - **Google's Photorealistic 3D Tiles API:** For immersive 3D terrain and environment rendering.
   - **OpenStreetMap (OSM) API:** For base map data, routing, and ground-truth comparison.

3. **Live Tracking & Data APIs (Dynamic Layers):**
   - **NORAD Satellite Data (Space-Track API):** Active satellite NORAD tracking and TLE data.
   - **OpenSky Network / ADS-B Exchange APIs:** Real-time commercial and military flight tracking.
   - **USGS Earthquake API:** Global live seismic activity.
   - **Austin Open Data:** Live CCTV Feed integrations.

## 🏗️ Current System Architecture Concept

The platform will act as a unified dashboard blending static and dynamic data layers:

- **Base Layer:** Google Photorealistic 3D Tiles / OSM Maps.
- **Infrastructure Layer:** Vector overlays containing the AI-extracted road networks (output from models like D-LinkNet or Sat2Graph).
- **Live Event Layers:** Points, lines, and heatmaps visualizing flights, satellites, and earthquakes updated in near real-time via websockets or polling mechanisms.

## 🛑 Current Status

- **Phase:** Robust SOTA Planning & Paper Ideation.
- **Completed:** Analyzed the research PDF, drafted the Study Checklist and `SOTA_Paper_Plan.md` containing the scoring metrics.
- **Next Immediate Action:** Finalize the architectural diagrams and await Docker installation so we can run AI-Researcher-AV1 for novel algorithm generation.

## 📝 Rules for Future Conversations

1. Always check this file first.
2. Update the "Current Status" and "Next Immediate Action" sections whenever a goal is accomplished or the project pivots.
3. Ensure no API keys are ever hardcoded in the application (use `.env`).
4. Keep the platform highly performant (handle dense geospatial data efficiently).
