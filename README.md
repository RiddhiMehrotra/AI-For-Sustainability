This project implements a complete geospatial machine‑learning pipeline for land‑use classification over the Delhi‑NCR region using Sentinel‑2 imagery and ESA WorldCover 2021 labels.

The dataset consists of:
Sentinel‑2 RGB patches (128×128 px, 10m resolution)
ESA WorldCover 2021 raster (cropped to Delhi‑NCR)
Shapefiles for Delhi‑Airshed and Delhi‑NCR regions
CSV files containing:
filtered patches (filtered_patches.csv)
assigned labels (train_labels.csv, test_labels.csv)

How to Run the Code
1. Install Requirements
pip install -r requirements.txt

3. Run Q1 (Spatial Filtering)
python src/q1_spatial.py

5. Run Q2 (Label Generation)
python src/q2_label.py

6. Run Q3 (Training & Evaluation)
python src/q3_training.py


I consulted an AI assistant(Claude) for clarifying concepts, improving code structure and debugging support during development. The implementation, model design, and analysis were completed by me.
