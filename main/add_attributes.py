import cv2
import sys
import os
import numpy as np
import pandas as pd

def apply_otsu(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholded_image = 255 - thresholded_image

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresholded_image, connectivity=8)

    # Ignore background (label 0) and get largest component
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.uint8(labels == largest_label) * 255
        result = cv2.bitwise_and(thresholded_image, thresholded_image, mask=mask)

    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    area = result.sum()/255

    mask_contour = np.zeros(image.shape, dtype=np.uint8)
    mask_contour = cv2.drawContours(mask_contour, contours, -1, 255, 1)
    perimeter = int(mask_contour.sum()/255)

    avg_contour = cv2.mean(image, mask=mask_contour)[0]
    avg_intensity = cv2.mean(image, mask=result)[0]    

    circularity = -1

    if perimeter > 0:
        circularity = (4 * 3.14 * area / (perimeter) ** 2)
    if perimeter < 5:
        elongation = -1
    else:
        ellipse = cv2.fitEllipse(contours[0])
        elongation = max(ellipse[1]) / min(ellipse[1]) if min(ellipse[1]) != 0 else 0

    return area, perimeter, avg_contour, avg_intensity, elongation

def add_attributes_to_dataframe(batch_name, dataframes_folder, images_folder, project_name):
    # Read the dataframe
    dataframe_path = os.path.join(dataframes_folder, f"{batch_name}_{project_name}.csv")
    images_folder = os.path.join(images_folder, batch_name, "samples")
    df = pd.read_csv(dataframe_path)

    areas = []
    perimeters = []
    avg_contours = []
    avg_intensities = []
    elongations = []

    for row in df.itertuples(index=False):
        image_name = row.names
        image_path = os.path.join(images_folder, image_name)
        
        area, perimeter, avg_contour, avg_intensity, elongation = apply_otsu(image_path)
        areas.append(area)
        perimeters.append(perimeter)
        avg_contours.append(avg_contour)
        avg_intensities.append(avg_intensity)  
        elongations.append(elongation)
    
#    if 'area' not in df.columns:    
    df['area'] = areas

#    if 'perimeter' not in df.columns:
    df['perimeter'] = perimeters
    df['circularity'] = (4 * 3.14 * df['area']) / (df['perimeter'] ** 2)

#    if 'avg_contour' not in df.columns:
    df['avg_contour'] = avg_contours

#    if 'avg_intensitity' not in df.columns:
    df['avg_intensity'] = avg_intensities   

#    if 'elongation' not in df.columns:
    df['elongation'] = elongations 

    # Save the updated dataframe
    df.to_csv(dataframe_path, index=False)


if len(sys.argv) > 1:
    project_name = sys.argv[1]
else:
    print("Usage: python add_attributes.py <project_name>")
    sys.exit(1)

dataframes_folder = os.path.join("main", "assets", project_name, "dataframes")
images_folder = os.path.join("main", "assets", project_name, "images")

list_batches = os.listdir(images_folder)
list_batches.sort()

for batch_name in list_batches[:10]:
    # Check if the batch folder is a directory
    if os.path.isdir(os.path.join(images_folder, batch_name)):
        print(f"Adding attributes to: {batch_name}")
        add_attributes_to_dataframe(batch_name, dataframes_folder, images_folder, project_name)
    else:
        print(f"Skipping non-directory: {batch_name}")


