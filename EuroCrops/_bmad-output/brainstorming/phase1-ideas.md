# EuroCrops Brainstorming — Phase 1 Ideas
**Session:** 2026-03-25 | **Technique:** What If Scenarios
**Topic:** Geospatial project scope — data engineering + CV/image segmentation + geospatial challenges

---

## Data Engineering

**[DE #1]: The Migration Tracker**
Given EuroCrops parcel boundaries as fixed anchors, build a system that ingests multi-year satellite imagery and tracks how land cover evolves inside each polygon over time — crop transitions, vegetation growth/shrinkage, fallow cycles.

**[DE #2]: The Irregular Time Series Problem**
Satellites don't deliver clean, evenly-spaced images — clouds create gaps. Build a pipeline that handles missing timesteps via masking, interpolation, and gap-filling. Model data as a 4D array (x, y, time, band) in zarr + xarray, normalized into consistent bi-weekly composites.

**[DE #3]: The 4D Pipeline Problem**
End-to-end pipeline: STAC catalog → odc-stac clip to parcel boundaries (GeoParquet) → cloud mask → temporal alignment → zarr temporal cube → PostGIS join with vector labels → PyTorch DataLoader. Each arrow is a format conversion problem worth understanding.

**[DE #4]: The Small Parcel Pipeline**
Design the entire pipeline around the constraint that Belgian parcels are ~20×10 pixels at Sentinel-2 10m resolution. Apply a 1-pixel inward buffer on every clip to eliminate edge pixel contamination. Makes every GDAL parameter choice visible.

**[DE #5]: Uncertainty-Aware Gap Filling**
Don't just drop cloudy observations — track *confidence* of each pixel through time as a separate channel. A pixel seen 8× in spring = high confidence; once in summer = low. Feed this confidence map as an additional model input channel.

**[DE #6]: The Weather Fusion Join**
ERA5 reanalysis data lives on a 0.25° grid (~27km). Joining it to 2-hectare parcels requires spatial downscaling + temporal alignment to Sentinel-2 observation timestamps. Build as a PostGIS + xarray pipeline attaching weather scalars (rainfall, temp, solar radiation, soil moisture) to each zarr parcel cube.

**[DE #7]: The Format Zoo — Why Each One Exists**
- **COG (Cloud Optimized GeoTIFF):** HTTP range-request just the pixels you need from S3. Raw input from STAC.
- **zarr:** N-dimensional chunked arrays. Your processed temporal cube — chunk by parcel for efficient per-parcel time-series reads.
- **netCDF / HDF5:** Scientific standard, 30 years old. ERA5 weather data arrives in this format.
- **GeoParquet:** Columnar vector data. EuroCrops boundaries ship in this format. DuckDB reads it natively with SQL.
- **PostGIS:** Relational spatial database for complex joins — parcel geometry + predicted state + confidence + crop label.

**[DE #8]: The Format Pipeline Is the Project**
```
STAC Catalog (COG)
    → odc-stac clip to parcel boundaries (GeoParquet)
    → cloud mask + temporal align
    → zarr temporal cube  ←→  PostGIS (vector + labels)
    +  ERA5 (netCDF) fusion
    → PyTorch DataLoader → Model
    → Predictions back into PostGIS → export as COG
```
Each arrow is a format conversion with a concrete reason. Understanding the *why* at each step is the core data engineering lesson.

**[DE #9]: ERA5 Is Already on GCS as Zarr**
ERA5 is publicly available on Google Cloud Storage (`gs://gcp-public-data-arco-era5`) in zarr format. Stream just the Belgium bounding box + years + variables using xarray + fsspec — no CDS API queue, no account. ~200–400MB for Belgium, 5 years, 6–8 variables at daily aggregation.

**[DE #10]: ERA5-Land Instead of ERA5**
ERA5-Land is higher resolution (0.1° ≈ 11km vs 0.25° ≈ 27km), focused on land-surface variables. Soil moisture from ERA5-Land is a strong predictor of crop stress. Slightly larger (~1–2GB for Belgium/5yr) but physically more meaningful for agriculture.

**[DE #11]: Docker Compose as the Local Stack**
Full local environment as `docker-compose.yml`: PostGIS container, processing container (GDAL + odc-stac + xarray + PyTorch), zarr volume mount. One `docker compose up` = reproducible geospatial data engineering environment. Solves GDAL version hell and PROJ mismatches.

---

## Computer Vision / ML

**[CV #1]: Temporal Signature Classification (SITS)**
Feed a *stack* of images over time (e.g. 24 Sentinel-2 passes/year) per field. The model learns that each crop type has a unique NDVI trajectory — wheat greens in spring, yellows in summer, harvested in fall. Single images can't distinguish crops that look identical in July; their temporal fingerprint is unique.

**[CV #2]: 3D ConvNets / Video Transformers on Fields**
Treat the temporal image stack as a video. Apply video understanding architectures (3D convolutions, TimeSformer, ViViT) to detect not just *what* a field is but *when* things happen — planting date, peak growth, harvest event.

**[CV #3]: Prediction Targets at t+x**
Four options for what to predict at t+x:
- **A — Segmentation mask:** What crop type will this field be? (classification, discrete)
- **B — Raw image:** What will this field look like? (generative, continuous)
- **C — NDVI scalar:** What will the vegetation index be? (regression, most practical)
- **D — Change mask:** Which pixels will have changed? (binary/multi-class)
Start with A, B, C inside the boundary. D and boundary morphology deferred to Phase 2.

**[CV #4]: Architecture — ConvLSTM / 3D-UNet Encoder-Decoder**
ConvLSTM or 3D-UNet encoder processes the temporal image stack spatially and temporally → decoder outputs predicted state at t+x. Irregular timesteps encoded as positional embeddings (time delta between observations), borrowed from NLP transformers applied to satellite data.

---

## VLM

**[VLM #1]: "What Changed Here?" as a Language Problem**
Feed before/after image pair (or temporal strip of thumbnails) to a VLM and ask it to describe what happened. Turns change detection from a pixel-level binary mask into a natural language narrative — interpretable by a non-expert.

**[VLM #2]: The Interpretability Layer**
After ConvLSTM predicts vegetation state at t+x, pass (input sequence + prediction) to a VLM and ask: "Why did you predict this? What in the historical sequence suggests this trajectory?" VLM becomes the model's interpreter — turning dense tensor predictions into human-readable explanations.

---

## MLOps

**[MLOps #1]: Serving Geospatial Predictions at Parcel Scale**
Inference pipeline runs over thousands of parcels on new Sentinel-2 passes every 5 days — a batch prediction DAG, not a REST API. Options: Airflow/Prefect DAG, or cron + Python. Tracks which parcels are "prediction-fresh" vs. "stale" based on cloud coverage of latest pass.

**[MLOps #2]: Format-Aware Serving**
Inference outputs predictions per parcel stored back into PostGIS as time-indexed rows, but also exported as COGs so downstream tools (QGIS, web maps) can consume them without touching the database. Input was COG, output is COG — the pipeline closes the loop.

**[MLOps #3]: The Scheduling Problem**
Every 5-day Sentinel-2 pass triggers: query STAC → clip parcels → update zarr → run inference → write predictions to PostGIS. Some parcels are cloudy → their zarr gets a masked timestep → inference skipped or uncertainty-flagged → scheduler tracks freshness. Stateful streaming over geospatial data.

---

## Geospatial Domain Challenges

**[Domain #1]: The Small Parcel Problem**
Belgian parcels average ~2 hectares = ~20×10 pixels at Sentinel-2 10m resolution. Every spatial model assumption breaks down at this scale. Forces thinking about sub-pixel accuracy, edge pixel contamination, and whether spatial convolutions make sense — or whether collapsing to pixel-timeseries per parcel is better.

**[Domain #2]: The Cloud Catastrophe**
Belgium gets ~150 cloudy days/year. During critical crop growth windows (April–June), 60–70% of observations may be lost. Model must reason about what it *doesn't know*. Makes uncertainty quantification essential — a confident prediction over missing data is dangerous in production.

**[Domain #3]: The Multi-Resolution Spatial Join**
ERA5 (0.1°–0.25° grid) joined to 2-hectare parcels. Bilinear interpolation across 3 orders of magnitude in spatial scale, while aligning to Sentinel-2 timestamps. CRS mismatches, temporal resampling, and scale mismatch all happen simultaneously.

---

## Evaluation

**[Eval #1]: Beat the Right Baseline**
"Predict same as last observation" (persistence) is hard to beat for slow-changing vegetation. Real baseline = seasonal climatology: what does this field *typically* look like at this time of year? Separating "learned temporal dynamics" from "learned seasonal patterns" is a core evaluation question.

**[Eval #2]: Domain-Aware Metrics**
Standard MSE on pixel values is misleading geospatially. Use:
- NDVI MAE per parcel (agronomy-interpretable)
- SSIM if predicting images (spatial coherence)
- Crop type accuracy if predicting segmentation masks
- Calibration score on uncertainty estimates (are confidence intervals reliable?)

**[Eval #3]: The Cloud-Held-Out Test**
Hold out observations that were actually cloud-free during training, mask them as cloudy, measure how well the model fills those gaps. Tests temporal interpolation capability directly. Simple to implement; directly validates the uncertainty pipeline.

**[Eval #4]: Country Generalization Test**
Train on Belgium, evaluate on a second EuroCrops country (Denmark, Slovenia) without retraining. Track metric degradation. If model collapses on a new country, it learned "Belgian agriculture," not "vegetation dynamics" — a meaningful scientific finding.

---

## Stretch / Connections

**[Stretch #1]: Robotics Bridge — Traversability Maps**
A vegetation state predictor at t+x is also a terrain state forecast for autonomous navigation. Satellite-derived field state (tall crop, fallow, wet soil, stubble) feeds into traversability maps for agricultural robots. Direct bridge between geospatial and robotics interests.

**[Stretch #2]: What If Trained on Belgium, Deployed on Another Country?**
Domain adaptation in geospatial. Field sizes triple from Belgium to Romania; crop diversity changes. Zero-shot geographic generalization as a core evaluation challenge. Also forces a hard question: what did the model actually learn?

**[Stretch #3]: The Pipeline as the Artifact**
A clean, reproducible, cloud-masked, temporally-aligned zarr cube of Sentinel-2 time-series clipped to EuroCrops parcels doesn't exist as a well-documented open artifact. Publishing the pipeline (and contributing pieces to odc-stac) may be more impactful than the model itself.

---

## Scoping Decision

**Inside-Before-Outside:**
Phase 1 scope = A, B, C (what happens *inside* fixed parcel boundaries — vegetation/crop state prediction).
Phase 2 scope (deferred) = boundary morphology, inter-field relationships, land fragmentation, boundary migration over time.
