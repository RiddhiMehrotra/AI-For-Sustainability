import glob
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

gdf= gpd.read_file("data/raw/delhi_ncr/delhi_ncr_region.geojson")
gdf_region= gdf
print(gdf_region.head())
print(gdf_region.crs)

gdf_region.to_crs(epsg=4326, inplace=True)
print(gdf_region.crs)
gdf_region_utm= gdf_region.to_crs(epsg=32644)
print(gdf_region_utm.total_bounds)

region_geom_utm= gdf_region_utm.geometry.union_all()

minx, miny, maxx, maxy= gdf_region_utm.total_bounds
grid_cells= []
for x in range(int(minx), int(maxx), 60000):
    for y in range(int(miny), int(maxy), 60000):
        cell= Polygon([(x, y), (x+60000, y), (x+60000, y+60000), (x, y+60000)])
        grid_cells.append(cell)
grid= gpd.GeoDataFrame({'geometry': grid_cells}, crs="EPSG:32644")
grid= grid[grid.intersects(region_geom_utm)]

patch_folder= Path("data/raw/sentinel_patches")
patch_files= list(patch_folder.glob("*.png"))
print("Total images before filtering:", len(patch_files))


##extract center LAT/LON from Files
records=[]
geoms=[]
for p in patch_files:
    stem= p.stem
    parts= stem.split("_")
    lat= float(parts[0])
    lon= float(parts[1])

    records.append({"filename": p.name, "lat": lat, "lon": lon})
    geoms.append(Point(lon, lat))

points_gdf= gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:4326")
points_utm= points_gdf.to_crs(epsg=32644)


mask = points_utm.within(region_geom_utm)
filtered_points_utm = points_utm[mask].copy()

print("Total images after filtering:", len(filtered_points_utm))

out_dir= Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

filtered_points_utm[["filename", "lat", "lon"]].to_csv(out_dir / "filtered_patches.csv", index=False)
print("Saved:", out_dir / "filtered_patches.csv")

#plot boundary
fig, ax = plt.subplots(figsize=(10, 10))
gdf_region_utm.boundary.plot(ax=ax, edgecolor='red', linewidth=1, facecolor='none')
grid.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
filtered_points_utm.plot(ax=ax, markersize=3, color='blue', label='Filtered Patches')

plt.title("Delhi-NCR shapefile with Filtered Patches")
plt.tight_layout()
plt.show()