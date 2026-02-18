import zipfile
import os

def zip_directory(folder_path, output_path):
    count = 0
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            # Exclude __pycache__
            if '__pycache__' in dirs:
                dirs.remove('__pycache__')
            
            for file in files:
                # Exclude the zip file itself and this script
                if file == os.path.basename(output_path) or file == 'create_submission_zip.py':
                    continue
                
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
                count += 1
    return count

source_dir = r"d:\multi\publication_package"
# Save the zip file in the parent directory to avoid recursive issues if running multiple times, 
# or just clearly exclude it (which I did). 
# Let's save it to d:\multi for easier access.
output_zip = r"d:\multi\Submission_Package_IEEE_TGRS_Sumawan_Final.zip"

print(f"Zipping content of {source_dir} to {output_zip}...")
try:
    count = zip_directory(source_dir, output_zip)
    print(f"Successfully created: {output_zip}")
    print(f"Total files added: {count}")
except Exception as e:
    print(f"Error: {e}")
