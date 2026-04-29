import os
import shutil

source_dir = "gambar_dataset"
target_dir = "dataset_clean"

os.makedirs(os.path.join(target_dir, "0"), exist_ok=True)
os.makedirs(os.path.join(target_dir, "1"), exist_ok=True)

for patient in os.listdir(source_dir):
    patient_path = os.path.join(source_dir, patient)

    if os.path.isdir(patient_path):
        for label in ["0", "1"]:
            label_path = os.path.join(patient_path, label)

            if os.path.exists(label_path):
                for file in os.listdir(label_path):
                    src = os.path.join(label_path, file)
                    dst = os.path.join(target_dir, label, file)

                    shutil.copy(src, dst)

print("Dataset berhasil dirapikan!")