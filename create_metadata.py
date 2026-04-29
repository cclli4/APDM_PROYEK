import os
import pandas as pd

label_df = pd.read_csv("calc_case_description_train_set.csv")

image_dir = "dataset_png"
data = []

for file in os.listdir(image_dir):
    if file.endswith(".png"):

        # ambil nama dasar
        base_name = file.split("_1_")[0]

        # cari yang mirip di CSV
        match = label_df[label_df["image file path"].str.contains(base_name, na=False)]

        if not match.empty:
            pathology = match.iloc[0]["pathology"]

            if pathology == "BENIGN":
                label = 0
            else:
                label = 1

            data.append([file, label])

df = pd.DataFrame(data, columns=["image", "label"])
df.to_csv("metadata_fixed.csv", index=False)

print("Metadata berhasil dibuat!")