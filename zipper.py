import os
import zipfile

zip_name = "nostalgia from server pc.zip"
current_dir = os.getcwd()

with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
    for foldername, subfolders, filenames in os.walk(current_dir):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)

            # Skip the zip file itself
            if os.path.abspath(file_path) == os.path.abspath(zip_name):
                continue

            # Preserve folder structure inside the zip
            arcname = os.path.relpath(file_path, current_dir)
            zipf.write(file_path, arcname)

print(f"Created zip archive: {zip_name}")
