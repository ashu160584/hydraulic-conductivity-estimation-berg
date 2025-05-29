"""
Hydraulic conductivity estimation and visualization using Ryd's formula.

This script reads groundwater well data from CSV files, calculates hydraulic conductivity
(K) based on Ryd's empirical formula, filters the data within a defined polygon area,
and produces cumulative distribution function (CDF) plots for different depth intervals.

Author: Ashutosh Singh
Date: 2024-05-29
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import urllib.request as urllib2
import requests
import zipfile
import io
import feedparser
from scipy.stats import norm, gmean

# === Constants ===
BASE_DIR = Path.cwd() / 'outputs'
INPUT_SHAPEFILE = Path.cwd() / 'data'
filename = 'vagkorsning_area.shp'

DIRS = {
    'xml': BASE_DIR / 'XML',
    'zip': BASE_DIR / 'SHP' / 'ZIP',
    'csv': BASE_DIR / 'CSV',
    'xlsx': BASE_DIR / 'XLSX',
    'rst': BASE_DIR / 'RST',
    'shp': BASE_DIR / 'SHP',
    'gpkg': BASE_DIR / 'GPKG',
    'png': BASE_DIR / 'PNG',
    'input_shapefile': INPUT_SHAPEFILE / 'shapefile_example',
    'layer': 'brunnar',
}
#URL list
urlsgu = []
urlsgu.append("https://resource.sgu.se/data/oppnadata/brunnar/brunnar.zip")

# Create the output folders (skip non-path entries)
for key, path in DIRS.items():
    if isinstance(path, Path):
        path.mkdir(parents=True, exist_ok=True)
for url in urlsgu:
    # Step 2: Extract ZIP into memory
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Find the GPKG file
        gpkg_filename = [name for name in z.namelist() if name.endswith(".gpkg")][0]
        # Extract GPKG to required directory
        z.extract(gpkg_filename, DIRS['gpkg'])
        # Build the full path to the extracted GPKG file
        extracted_gpkg_path = DIRS['gpkg'] / gpkg_filename

# Step 3: Read the GPKG file using geopandas
print("📂 Extracted GPKG path:", extracted_gpkg_path)
gdf = gpd.read_file(extracted_gpkg_path, layer=DIRS['layer'])
gdf = gdf.to_crs(3011)

# 4. Convert geometry to latitude and longitude
gdf["x"] = gdf.e
gdf["y"] = gdf.n

# 5. Drop the original geometry column
gdf = gdf.drop(columns="geometry")

extracted_csv_path = DIRS['csv'] / "brunnar.csv"
gdf.to_csv(extracted_csv_path, index=False)

print(f"CSV saved to: {DIRS['csv']}")
############################################




# === Functions ===

def load_combined_csv(csv_dir: Path) -> pd.DataFrame:
    """
    Reads and combines all CSV files in the given directory.

    Args:
        csv_dir (str): Path to the directory containing CSV files.

    Returns:
        pd.DataFrame: Combined DataFrame with required columns, cleaned of missing coordinates or flow.
    """
    combined_df = pd.DataFrame()
    for filepath in csv_dir.iterdir():
        if filepath.suffix.lower() == '.csv':
            df = pd.read_csv(filepath, delimiter=',', encoding='ISO-8859-1', on_bad_lines='skip')
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    combined_df.dropna(subset=['n', 'e', 'kapacitet'], inplace=True)
    return combined_df

def calculate_K(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates hydraulic conductivity K using Ryd's formula.

    K = 0.0756 * (Q / 1000 / 3600)^1.0255 / (L/t)
    where Q is flow in l/h, and L/t is the difference between total well depth and groundwater level.

    Args:
        df (pd.DataFrame): Input DataFrame containing at least 'kapacitet', 'totaldjup', and 'grundvattenniva'.

    Returns:
        pd.DataFrame: Filtered DataFrame with a new column 'K'.
    """
    df['grundvattenniva'].fillna(df['jorddjup'], inplace=True)
    df = df[df['totaldjup'] != df['grundvattenniva']]
    df['K'] = 0.0756 * (df['kapacitet'] / 1000 / 3600) ** 1.0255 / (df['totaldjup'] - df['grundvattenniva'])
    df = df[(df['K'].notna()) & (df['K'] > 0) & np.isfinite(df['K'])]
    return df


def clip_to_polygon(gdf: gpd.GeoDataFrame, shapefile_path: Path) -> gpd.GeoDataFrame:
    """
    Clips a GeoDataFrame of points to the boundary of a given shapefile polygon.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame with well data and geometry.
        shapefile_path (str): Path to the shapefile containing the area of interest.

    Returns:
        gpd.GeoDataFrame: Clipped GeoDataFrame containing only wells inside the polygon.
    """

    area = gpd.read_file(shapefile_path).to_crs(epsg=3011)
    gdf = gdf.to_crs(epsg=3011)
    result = gpd.sjoin(gdf, area, how='inner', predicate='intersects')
    return result


def plot_cdf_by_intervals(result_gdf: gpd.GeoDataFrame, intervals: list[tuple[int, int]], titles: list[str],
                          plot_title: str, output_dir: str) -> None:
    """
    Plots cumulative distribution functions (CDF) of K for multiple depth intervals.

    Also saves the result as a PNG file and an Excel workbook with one sheet per interval.

    Args:
        result_gdf (gpd.GeoDataFrame): GeoDataFrame containing well data with K-values.
        intervals (list[tuple[int, int]]): List of (min_depth, max_depth) tuples.
        titles (list[str]): Corresponding labels for each depth interval.
        plot_title (str): Title for the plot and output files.
        output_dir (str): Base directory where output PNG and Excel files will be saved.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green', 'black', 'blue']
    all_data = []

    for i, interval in enumerate(intervals):
        djup_min, djup_max = interval
        title = titles[i]
        subset = result_gdf[(result_gdf['totaldjup'] >= djup_min) & (result_gdf['totaldjup'] <= djup_max)]
        K_vals = np.sort(subset['K'].dropna().values)
        K_vals = K_vals[K_vals > 0]

        if len(K_vals) == 0:
            continue

        cdf_values = np.arange(1, len(K_vals) + 1) / len(K_vals)
        log_K = -np.log(K_vals)
        mu, sigma = np.mean(log_K), np.std(log_K)
        inv_cdf = np.arange(1, 1000) / 1000
        normal_quantiles = norm.ppf(1 - inv_cdf, loc=mu, scale=sigma)
        modeled_K = np.exp(-normal_quantiles)

        plt.semilogx(K_vals, cdf_values, '.', color=colors[i % len(colors)], label=f'{title} (K3D={gmean(K_vals)*np.exp(sigma**2/6):.2e})')
        plt.semilogx(modeled_K, inv_cdf, '-', color=colors[i % len(colors)])

        all_data.append((interval, subset))

    plt.title(plot_title)
    plt.xlabel('K (m/s)')
    plt.ylabel('p(K < Kn)')
    plt.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
    plt.ylim(0, 1)
    plt.xlim(1e-9, 1e-4)
    plt.legend()
    plt.savefig(DIRS['png']/ f'{plot_title}.png', dpi=300, bbox_inches='tight')
    plt.show()

    with pd.ExcelWriter(DIRS['xlsx']/ f'brunnar_{plot_title}.xlsx', engine='xlsxwriter') as writer:
        for (djup_min, djup_max), df in all_data:
            sheet_name = f'{djup_min}-{djup_max}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        result_gdf.to_excel(writer, sheet_name='samtliga brunnar', index=False)


# === Main ===
if __name__ == '__main__':
    # Load data
    df = load_combined_csv(DIRS['csv'])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.e, df.n), crs='EPSG:3006')

    # Clip and calculate
    result_gdf = clip_to_polygon(gdf, DIRS['input_shapefile'] / filename)
    result_gdf = calculate_K(result_gdf)

    gdf = gpd.GeoDataFrame(result_gdf, 
                       geometry=gpd.points_from_xy(result_gdf.e, result_gdf.n))
    gdf.set_crs('epsg:3011')

    gdf.to_file(DIRS['shp'] / 'brunnarinnanfor.shp')

    # Plot all data
    plot_cdf_by_intervals(
        result_gdf,
        intervals=[(0, result_gdf['totaldjup'].max())],
        titles=['Totaldjup'],
        plot_title='kumulativ fördelningsfunktion av K för samtliga brunnar',
        output_dir=BASE_DIR
    )

    # Plot by depth intervals
    plot_cdf_by_intervals(
        result_gdf,
        intervals=[(0, 50), (50, 100), (100, result_gdf['totaldjup'].max())],
        titles=['Totaldjup <= 50m', '50m < Totaldjup <= 100m', 'Totaldjup > 100m'],
        plot_title='kumulativ fördelningsfunktion av K för djupintervall',
        output_dir=BASE_DIR
    )