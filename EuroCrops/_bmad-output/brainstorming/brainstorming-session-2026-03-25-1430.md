---
stepsCompleted: [1, 2]
inputDocuments: []
session_topic: 'EuroCrops project scope — data engineering + CV/image segmentation + geospatial challenges'
session_goals: 'Define a cohesive project that covers PostGIS/geospatial data engineering, image segmentation model training and deployment (+ VLM), and surfaces real geospatial industry challenges. Identify datasets and tech stack.'
selected_approach: 'all-four-progressive-random-ai-user'
techniques_used: []
ideas_generated: []
context_file: ''
---

# Brainstorming Session Results

**Facilitator:** Emulie
**Date:** 2026-03-25

## Session Overview

**Topic:** EuroCrops project scope — data engineering + CV/image segmentation + geospatial challenges
**Goals:** Define a cohesive project that covers PostGIS/geospatial data engineering, image segmentation model training and deployment (+ VLM), and surfaces real geospatial industry challenges. Identify datasets and tech stack.

### Session Setup

Emulie is exploring how to design a learning project around EuroCrops that is genuinely challenging across three dimensions:
1. **Data engineering**: PostGIS, geospatial file formats (parquet, zarr, HDF, netCDF, COG, GeoTIFF), pipelines
2. **Computer vision / ML**: Image segmentation model training + deployment, VLM exploration
3. **Geospatial domain understanding**: Real friction points in the field (projections, cloud-native access, labeling at scale, etc.)

Secondary interest: connections to robotics (SLAM, lane detection, self-driving).

## Technique Selection

**Approach:** All Four — Progressive Flow → Random → AI-Recommended → User-Selected
**Journey Design:** Start broad and systematic, inject chaos, refine with intelligence, then personalize

**Progressive Techniques (Round 1 — Phase 4):**

- **Phase 1 - Expansive Exploration:** What If Scenarios — generate maximum project possibilities without constraints
- **Phase 2 - Pattern Recognition:** Mind Mapping — cluster ideas into themes across data eng / CV / domain challenges
- **Phase 3 - Idea Development:** SCAMPER Method — refine the strongest project concept(s)
- **Phase 4 - Action Planning:** Solution Matrix — dataset × tech stack × challenge grid for implementation planning

---

## Phase 2: Pattern Recognition — Mind Map

### Key Threads

**Thread 1 — Geospatial challenges are the connective tissue, not a separate concern:**
- Small parcel (20×10px) → affects architecture + pipeline
- Cloud catastrophe → affects pipeline + model + evaluation
- Multi-resolution join → affects ERA5 fusion + PostGIS design

**Thread 2 — Two distinct spines share one hinge (zarr):**
- SPINE A (Data Engineering): COG → clip → mask → zarr → PostGIS → DAG → COG out
- SPINE B (ML): zarr + ERA5 → ConvLSTM → predict t+x → evaluate → VLM interpret
- zarr is where data engineering ends and ML begins

**Thread 3 — Natural MVP layering:**

**V1 (Months 1–2): Pipeline Spine**
- STAC → zarr cube (shape: n_parcels × n_timesteps × n_bands × H × W)
- Cloud masking + confidence channel + temporal alignment
- PostGIS as operational backbone, Docker Compose local stack
- Deliverable: reproducible temporal cube for Belgian parcels
- Geospatial lessons: formats, CRS, cloud masking, STAC, odc-stac contribution

**V2 (Months 3–4): Model Spine**
- One model, two output heads (Head A: segmentation mask at t+x, Head C: NDVI scalar at t+x)
- Irregular timesteps handled via time-delta positional embeddings — not image interpolation
- Start with Head C alone, add Head A once temporal backbone works
- Deliverable: working forecaster + baseline evaluation
- Geospatial lessons: small parcel problem, irregular timesteps, domain-aware metrics

**V3 (Months 5+): Robustness + Operations**
- ERA5 multimodal fusion → improves accuracy (weather as additional input channels)
- Prediction uncertainty → model outputs confidence intervals, not point estimates
- Scheduling DAG → 5-day inference cadence, parcel freshness tracking
- VLM interpretability layer
- Geospatial lessons: multi-resolution join, uncertainty quantification, MLOps

### Key Insight
The odc-stac open source contribution sits at step 1 of SPINE A. Friction found querying STAC for Belgian parcels is exactly where to contribute back — the project generates the contribution organically.

---

## Round 4: User-Selected — Morphological Analysis + Solution Matrix

### Morphological Analysis — Full Parameter Space

| Parameter             | Option A                       | Option B                  | Option C                      |
|-----------------------|--------------------------------|---------------------------|-------------------------------|
| **Geography**         | Spain first (sunny, simple)    | Belgium first (hard mode) | Both simultaneously           |
| **Sensor**            | Sentinel-2 only                | Landsat + Sentinel-2      | Sensor-agnostic abstraction   |
| **Supervision**       | Unsupervised (NDVI clustering) | Supervised (crop labels)  | Unsupervised → supervised     |
| **Prediction target** | NDVI scalar                    | Segmentation mask         | Both heads                    |
| **Architecture**      | ConvLSTM                       | 3D-UNet                   | Transformer + time embeddings |
| **Spatial scope**     | 5 parcels → 1,000              | Full country from start   | One commune (~500 parcels)    |
| **ERA5 fusion**       | No (V1 only)                   | Yes from V1               | Yes from V2 only              |
| **Storage**           | zarr + PostGIS                 | zarr only                 | PostGIS only                  |
| **VLM layer**         | No                             | V3 addition               | V2 addition                   |
| **Scheduling DAG**    | No                             | Simple cron               | Prefect/Airflow               |

### Selected Blueprint

| Parameter         | Choice    | Selection                               |
|-------------------|-----------|-----------------------------------------|
| Geography         | **A**     | Spain first (Andalusia)                 |
| Sensor            | **B**     | Landsat + Sentinel-2                    |
| Supervision       | **C**     | Unsupervised → supervised               |
| Prediction target | **C**     | Both heads (NDVI + mask)                |
| Architecture      | **A/B/C** | Decide in V2 — start with ConvLSTM      |
| Spatial scope     | **C**     | One Andalusian commune (~500 parcels)   |
| ERA5 fusion       | **C**     | Add in V2                               |
| Storage           | **A**     | zarr + PostGIS                          |
| VLM layer         | **B**     | V3 addition                             |
| Scheduling DAG    | **C**     | Prefect (simpler than Airflow for solo) |

**Notes:**
- Landsat + Sentinel-2: 50-year historical record at 30m resolution. Large Andalusian parcels handle 30m well. Forces sensor-agnostic abstraction from day one.
- Architecture deferred to V2 — start with ConvLSTM, switch if it underperforms.
- Prefect recommended over Airflow for solo local projects.

### Why Spain Before Belgium

| Dimension                    | Belgium                                  | Andalusia (Spain)                   |
|------------------------------|------------------------------------------|-------------------------------------|
| Cloud cover                  | ~150 cloudy days/yr → ~30–40 valid obs   | ~70 days → ~60–70 valid obs         |
| Parcel size (Sentinel-2 10m) | ~20×10 pixels                            | ~100×100+ pixels                    |
| Parcel size (Landsat 30m)    | ~7×3 pixels                              | ~30×30+ pixels                      |
| Crop diversity               | High — sugar beet, potato, chicory, flax | Low — olive, sunflower, cereal      |
| Landscape                    | Dense mosaic, severe edge mixing         | Large continuous agricultural zones |
| Spring window (critical)     | Often fully clouded                      | Mostly clear                        |

Belgium maximizes every hard problem simultaneously. Andalusia is hard enough to be real, forgiving enough to learn on. Start in Spain so when you hit a bug you know it's your code, not a bad data week.

### Solution Matrix

| Challenge                  | V1 Solution                                     | V2 Solution                                        | V3 Solution                                         |
|----------------------------|-------------------------------------------------|----------------------------------------------------|-----------------------------------------------------|
| **Multi-sensor alignment** | Sentinel-2 only, build sensor abstraction layer | Add Landsat via same abstraction                   | Validate both sensors produce consistent NDVI       |
| **Cloud masking**          | SCL band mask, drop cloudy pixels               | Add `is_imputed` confidence channel                | Uncertainty-aware prediction intervals              |
| **CRS mismatches**         | Assert CRS equality before every spatial op     | Automated reprojection with logged parameters      | Pipeline test suite for spatial alignment           |
| **Small parcels**          | 1-pixel inward buffer on all clips              | Experiment with pixel vs. parcel-level aggregation | Evaluate spatial vs. non-spatial architectures      |
| **Irregular timesteps**    | Bi-weekly composite with gap flags              | Time-delta positional embeddings in model          | Evaluate impact of imputation strategy on accuracy  |
| **ERA5 join**              | Not yet                                         | Spatial downscaling + timestamp alignment          | Soil moisture + temperature as model input channels |
| **Supervision**            | Unsupervised NDVI trajectory clustering         | Add EuroCrops crop labels, name the clusters       | Multi-task head (NDVI + crop type simultaneously)   |
| **Scale**                  | 500 Andalusian parcels on MPS                   | Full Andalusia on MPS / small GCP                  | Belgium on GCP — deliberate hard mode               |
| **Reproducibility**        | Docker Compose, pinned STAC collection version  | zarr metadata logs every pipeline parameter        | CI test on fresh clone                              |
| **Open source**            | Use odc-stac, note friction points              | Contribute fix/improvement back                    | Document pipeline as reusable template              |

---

## Final Session Summary

### The Project in One Sentence
Build a geospatial pipeline that ingests Landsat + Sentinel-2 time-series for EuroCrops parcels in Andalusia, tracks vegetation state over time, and predicts it at t+x — using every engineering decision as a lesson in real geospatial challenges.

### The Real Goal
Geospatial understanding. Data engineering and CV/ML are vehicles, not destinations.

### Project Blueprint

**V1 — Pipeline Spine** *(Spain, 500 parcels, MPS)*
- Tools: DuckDB + GeoParquet → odc-stac → rasterio/GDAL → zarr + xarray → PostGIS + Docker
- Deliverable: reproducible temporal NDVI cube for one Andalusian commune
- Learning: formats, CRS, cloud masking, STAC, sensor abstraction, spatial joins
- Start unsupervised — no crop labels yet

**V2 — Model Spine** *(Spain, full Andalusia, MPS + GCP)*
- Tools: PyTorch ConvLSTM (start) → two heads (NDVI scalar + segmentation mask)
- Add: ERA5 weather fusion, `is_imputed` confidence channel, time-delta embeddings
- Add: EuroCrops crop labels to name the unsupervised clusters
- Learning: temporal models, irregular timesteps, multi-modal fusion, domain-aware evaluation

**V3 — Robustness + Belgium** *(Belgium, GCP, full stack)*
- Add: uncertainty quantification, Prefect scheduling DAG, VLM interpretability
- Switch geography to Belgium — deliberately experience the cloud catastrophe, small parcels, crop diversity
- Open source: contribute odc-stac fix, publish pipeline as reusable template

### Key Design Decisions
- Start in Andalusia (Spain) — forgiving data environment for learning
- Start unsupervised — NDVI clustering before crop classification
- Pipeline is the V1 deliverable — model just validates it works
- Sensor-agnostic abstraction from day one (Landsat + Sentinel-2)
- 5 parcels first, then 500, then scale — never more complexity than needed
- Prefect over Airflow for solo project
- Stream raw data (public cloud, free) — store only processed zarr locally (~15–20GB)

### Critical Design Checklist (from Reverse Brainstorming)
1. Every imputed timestep carries an explicit `is_imputed` flag
2. Every spatial operation asserts CRS equality before executing
3. Every temporal join documents its alignment tolerance explicitly
4. STAC collection version is pinned and logged in zarr metadata

### Ideas Generated: 50+ across data engineering, CV/ML, MLOps, geospatial domain challenges, evaluation, stretch goals
### Full idea list: `phase1-ideas.md`

---

## Round 3: AI-Recommended — Constraint Mapping + First Principles + Assumption Reversal

### Constraint Map

| Constraint | Reality                                                  | Implication                                                                |
|------------|----------------------------------------------------------|----------------------------------------------------------------------------|
| Compute    | MPS (Apple Silicon) local + GCP GPU when needed          | Prototype on subset (~500 parcels), scale to GCP for full runs             |
| Timeline   | Self-paced, no deadline                                  | Depth over speed — V1 as standalone complete artifact                      |
| Storage    | ~15–20GB local needed                                    | Stream raw Sentinel-2 COGs (free, public cloud), store only processed zarr |
| Skills     | Full stack is new (STAC, PostGIS, zarr, PyTorch, Docker) | Sequential introduction — one tool at a time                               |

**Sequential skill-building order:**
1. GeoParquet + DuckDB → explore EuroCrops, learn vector data
2. STAC + odc-stac → query + clip imagery, open source contribution angle
3. zarr + xarray → build temporal cube for 5 parcels first
4. PostGIS + Docker → spatial database, link vector to raster
5. PyTorch ConvLSTM → train on 500 parcels on MPS
6. GCP + full dataset → scale what already works

### First Principles

**The irreducible core:** "Build a system that turns free, repeat satellite observations of labeled fields into a temporally-aware prediction — and understand every engineering decision that makes that possible."

**The real north star:** Geospatial understanding is the destination. Data engineering and CV/ML are vehicles.

**Key implication:** Choose approaches that expose friction, not hide it. Use rasterio directly before odc-stac. Reproject manually with GDAL before letting a library do it silently. The model doesn't need to be good — it needs to surface real problems.

**Reframed V1 goal:** Build a pipeline that fails in instructive ways. Every error is the curriculum.

**Minimum honest project:** 5 parcels × 12 months × 1 band (NDVI) → ~2MB processed zarr. Understand every line of code. Scale only after full comprehension.

**Data sizes:**
- 5 parcels: ~2MB | 100 parcels: ~40MB | 1,000 parcels: ~400MB | All Belgium: ~2–3GB

### Assumption Reversals

**[Assumption #1 reversed]: Pipeline is the deliverable, model is the validator**
The hard part is the data, not the model. Design V1 so the pipeline is what you're proud of. V2 (model) just proves V1 works. Clean pipeline → any model works. Broken pipeline → no model saves it.

**[Assumption #2 reversed]: Start unsupervised, add labels in V2**
V1 doesn't need EuroCrops labels. Build the temporal NDVI pipeline, cluster trajectories unsupervised, visualize in QGIS. Labels come in V2 to name the clusters.

**[Assumption #3 reversed]: Start in Spain, move to Belgium deliberately** ✓ *Selected*
Belgium has 150 cloudy days/year, tiny parcels, complex rotations — worst learning environment. Andalusia (Spain, in EuroCrops) has 300+ sunny days, large flat parcels, simple rotations. Learn the tools on clean data, then bring hard lessons to Belgium in V2 as a known, studied challenge.

---

## Round 2: Random Techniques — Alien Anthropologist + Reverse Brainstorming

### Alien Anthropologist

**[Alien #1]: The Human Decision Layer**
*Concept:* Crop transitions are partially driven by farmer economic decisions — grain prices, EU CAP subsidies, policy changes — none of which are in the input data. Adding macro signals (EU crop price indices, CAP policy change flags) as scalar time-series alongside ERA5 could capture the economic forcing that drives collective behavior.
*Novelty:* Turns a vegetation forecasting model into a socio-economic + environmental model. The geospatial data becomes evidence of human behavior, not just physical state. If EU subsidy rules change, a model trained on 2017–2022 will be confidently wrong without this signal.

**[Alien #2]: The Independence Assumption**
*Concept:* The model treats each parcel independently. But neighboring parcels share soil types, microclimates, and the same local farmer community. A field left fallow next door often signals something about your own field. A graph neural network where parcels are nodes connected by adjacency could capture neighborhood effects.
*Novelty:* Challenges the parcel-as-independent-unit assumption. Genuinely different architecture from ConvLSTM — opens the door to spatial graph modeling as an alternative or complement.

### Reverse Brainstorming — Failure Modes → Design Requirements

**[Failure #1]: The Silent Mean**
*How it fails:* Missing timesteps (e.g. flooded field all of July) are filled with band mean across all parcels/years. Model sees "normal July values," predicts normal harvest. Field was lost. No warning anywhere.
*Design requirement:* Every imputed timestep carries an explicit `is_imputed=True` flag as a separate channel in zarr. Model sees it. Evaluation reports "% imputed inputs" per prediction. Predictions built on >40% imputed timesteps get a forced low-confidence flag.

**[Failure #2]: The Silent CRS Mismatch**
*How it fails:* Parcel geometries stored in WGS84, imagery clipped in UTM without reprojection. Clips look correct visually but are 15–30m off. Model trains on systematically misaligned data, learns to ignore boundary pixels.
*Design requirement:* Every spatial operation asserts CRS equality before executing. Pipeline test checks that the center of every clipped raster falls within the source polygon.

**[Failure #3]: The Temporal Ghost Join**
*How it fails:* Sentinel-2 captures at ~10:30 AM local. ERA5 joined naively by date. At daily granularity this is minor noise; at hourly it's a systematic mismatch — joining morning observations to midnight values.
*Design requirement:* Temporal join tolerance defined explicitly (e.g. ±6 hours of overpass time). Every cross-dataset join documents its alignment assumption. Tolerance is a configurable parameter, not a hardcoded default.

**[Failure #4]: The Dataset Version Drift**
*How it fails:* zarr cube built from Sentinel-2 Collection 0. New inference data comes from Collection 1 (improved atmospheric correction). Model sees systematically different pixel values for the same physical conditions, degrades silently.
*Design requirement:* Sentinel-2 collection version pinned as a hard parameter in STAC queries. Logged in zarr metadata. Pipeline assertion that training and inference data share the same collection version.

**Common root cause across all failures:** Silent assumptions with no assertions, no flags, no tests.

**Design Checklist:**
1. Every imputed timestep carries an explicit `is_imputed` flag
2. Every spatial operation asserts CRS equality before executing
3. Every temporal join documents its alignment tolerance explicitly
4. STAC collection version is pinned and logged in zarr metadata
