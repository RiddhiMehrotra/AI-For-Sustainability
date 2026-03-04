import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import rasterio
from rasterio.windows import Window, transform
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

filtered_patches= pd.read_csv("data/processed/filtered_patches.csv")
for idx, row in filtered_patches.iterrows():
    filename= row["filename"]
    stem = Path(filename).stem
    p = stem.split("_")
    lat= float(p[0])
    lon= float(p[1])

    with rasterio.open("data/raw/land_cover.tif") as src:
        transform = src.transform
        crs = src.crs
        
        row, col = src.index(lon, lat)
        row_start= row - 64
        col_start = col -64
        window = Window(col_start, row_start, 128, 128)
        if  row_start < 0 or col_start < 0 or row_start + 128 > src.height or col_start + 128 > src.width:
            print(f"Skipping {filename} due to out-of-bounds window.")
            continue
        patch_data = src.read(1, window=window)
        patch_data = np.where(patch_data == 0, np.nan, patch_data)
        flat=patch_data.flatten()  # flatten 128x128 into 1D array
        flat= flat[~np.isnan(flat)] #to remove NaNs(invalid pixels)
        if flat.size == 0:
            print(f"Skipping {filename}: empty patch")
            continue
        mode_result=stats.mode(flat, keepdims=True)
        dominant_class=int(mode_result.mode[0])

class_map={
    50: "Built-up",
    40:"Cropland",
    80:"Water",
}
veg_classes=[10,20,30]
results =[]
if dominant_class in class_map:
    category=class_map[dominant_class]
elif dominant_class in veg_classes:
    category="Vegetation"
else:
    category="Other"
results.append({"filename": filename, "lat": lat, "lon": lon, "esa_class":dominant_class,"category": category})
label_df= pd.DataFrame(results)
label_df.to_csv("data/processed/labeled_patches.csv", index=False)
train_df, test_df=train_test_split(label_df, test_size=0.4, random_state=42, stratify=label_df["category"])
train_df.to_csv("data/processed/train_labels.csv", index=False)
test_df.to_csv("data/processed/test_labels.csv", index=False)
label_df['category'].value_counts().plot(kind='bar')
plt.title("Distribution of Land Cover Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()
