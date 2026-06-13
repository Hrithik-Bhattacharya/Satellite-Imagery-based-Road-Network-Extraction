import os

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from skimage.morphology import skeletonize


def rasterize_osm_centerlines(geotiff_path, shapefile_path, output_mask_path):
    """
    Converts OSM vector lines into a 1-pixel wide raster mask perfectly
    aligned with a geo-referenced satellite image.
    """
    # 1. Read the satellite image metadata (GPS coordinates, size, projection)
    with rasterio.open(geotiff_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        satellite_crs = src.crs

    # 2. Load the OpenStreetMap vector lines
    osm_vectors = gpd.read_file(shapefile_path)

    # 3. CRITICAL: Align the GPS projections.
    # If the OSM map and Satellite use different GPS formats, align them.
    if osm_vectors.crs != satellite_crs:
        osm_vectors = osm_vectors.to_crs(satellite_crs)

    # 4. Burn the vector lines into a blank pixel array (1-pixel wide)
    # We burn the lines as white (255) on a black (0) background
    shapes = ((geom, 255) for geom in osm_vectors.geometry if geom is not None)

    weak_mask = features.rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        # Background is black
        dtype="uint8",
        all_touched=True,
        # Ensures the line is completely connected
    )

    # 5. Save the generated weak label
    cv2.imwrite(output_mask_path, weak_mask)
    print(f"Weak label successfully saved to {output_mask_path}")


def simulate_weak_labels_deepglobe(input_mask_dir, output_weak_dir):
    """
    Takes standard thick pixel masks and skeletonizes them into 1-pixel centerlines
    to simulate weak labels for training.
    """
    if not os.path.exists(output_weak_dir):
        os.makedirs(output_weak_dir)

    masks = [f for f in os.listdir(input_mask_dir) if f.endswith("_mask.png")]

    for mask_name in masks:
        input_path = os.path.join(input_mask_dir, mask_name)
        output_path = os.path.join(output_weak_dir, mask_name)

        # 1. Load the thick DeepGlobe mask in grayscale
        thick_mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # 2. Convert to strict binary (0 and 1) for the skeletonize algorithm
        binary_mask = (thick_mask > 127).astype(np.uint8)

        # 3. Crush the road down to its absolute 1-pixel centerline
        skeleton = skeletonize(binary_mask)

        # 4. Convert back to image format (0 and 255) and save
        weak_label = (skeleton * 255).astype(np.uint8)
        cv2.imwrite(output_path, weak_label)

    print(f"Successfully generated {len(masks)} weak labels in {output_weak_dir}")


# --- Test the Skeletonizer ---
if __name__ == "__main__":
    # Ensure you pip install scikit-image before running this
    # Run this to generate your simulated weak training data
    simulate_weak_labels_deepglobe(
        input_mask_dir="../data/deepglobe/train",
        output_weak_dir="../data/osm_weak_labels",
    )
