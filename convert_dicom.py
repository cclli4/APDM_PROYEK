import os
import pydicom
import numpy as np
from PIL import Image

input_dir = "CBIS-DDSM"
output_dir = "dataset_png"

os.makedirs(output_dir, exist_ok=True)

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".dcm"):
            dcm_path = os.path.join(root, file)
            
            dicom = pydicom.dcmread(dcm_path)
            img = dicom.pixel_array

            # normalisasi
            img = (img - img.min()) / (img.max() - img.min())
            img = (img * 255).astype(np.uint8)

            img = Image.fromarray(img)

            # bikin nama file unik
            folder_name = os.path.basename(root)
            save_name = folder_name + "_" + file.replace(".dcm", ".png")

            save_path = os.path.join(output_dir, save_name)
            img.save(save_path)

print("Convert selesai!")