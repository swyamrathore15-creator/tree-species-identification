import os
import glob
import rasterio

BASE = r"D:\ML PROJECT\Tree Species Identification\data\raw\TreeSatAI"

species_folders = [
    "aerial_60m_abies_alba",
    "aerial_60m_fagus_sylvatica",
    "aerial_60m_picea_abies",
    "aerial_60m_pinus_sylvestris",
    "aerial_60m_quercus_petraea"
]

print("=" * 60)
print("TreeSatAI DATASET VERIFICATION")
print("=" * 60)

total = 0

for folder in species_folders:
    folder_path = os.path.join(BASE, folder)

    files = glob.glob(os.path.join(folder_path, "**", "*.tif"), recursive=True)

    print(f"\n{folder}")
    print(f"TIFF images found: {len(files)}")

    total += len(files)

    if files:
        sample = files[0]

        with rasterio.open(sample) as src:
            print(f"Sample file: {os.path.basename(sample)}")
            print(f"Width: {src.width}")
            print(f"Height: {src.height}")
            print(f"Bands: {src.count}")
            print(f"Data type: {src.dtypes}")
            print(f"CRS: {src.crs}")

print("\n" + "=" * 60)
print(f"TOTAL TIFF IMAGES: {total}")
print("=" * 60)