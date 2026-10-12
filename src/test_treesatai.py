import rasterio
import os

image_path = r"D:\ML PROJECT\datasets\TreeSatAI\aerial_60m_abies_alba\Abies_alba_2_188_WEFL_NLF.tif"

if not os.path.exists(image_path):
    print("Image not found!")
    print(image_path)
else:
    with rasterio.open(image_path) as src:
        print("Image opened successfully!")
        print("Width:", src.width)
        print("Height:", src.height)
        print("Number of bands:", src.count)
        print("Data type:", src.dtypes)
        print("Coordinate system:", src.crs)