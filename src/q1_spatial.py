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
minx, miny, maxx, maxy= gdf_region_utm.total_bounds
grid_cells= []
for x in range(int(minx), int(maxx), 60000):
    for y in range(int(miny), int(maxy), 60000):
        cell= Polygon([(x, y), (x+60000, y), (x+60000, y+60000), (x, y+60000)])
        grid_cells.append(cell)
grid= gpd.GeoDataFrame({'geometry': grid_cells})
grid.set_crs(epsg=32644, inplace=True)
fig, ax = plt.subplots(figsize=(10, 10))
gdf_region_utm.boundary.plot(ax=ax, edgecolor='red', facecolor='none')
grid.boundary.plot(ax=ax, edgecolor='black', facecolor='none')
plt.title('Grid Overlay on Delhi NCR Region')
print(plt.show())

