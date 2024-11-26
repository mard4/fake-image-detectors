import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

### Export images with the same structure but only from TrueFace_PreSocial
path = "/media/NAS/TrueFace/TrueFace"
export_mainpath = "/media/SSD_mmlab/martina.dangelo/fake-image-detectors/dataset/media/NAS/TrueFace/TrueFace"
max_images_per_folder = 100  # Maximum number of images to copy per subfolder

# Walk through the directory structure
for root, dirs, files in os.walk(path):
    # Determine the relative path from the root directory
    rel_path = os.path.relpath(root, path)
    
    # Create the equivalent path in the export directory
    export_path = os.path.join(export_mainpath, rel_path)
    os.makedirs(export_path, exist_ok=True)
    
    # Check if the current directory is part of TrueFace_PreSocial
    if "TrueFace_PreSocial" in root:
        image_count = 0  # Initialize image counter for the current subfolder
        for file_name in files:
            if image_count >= max_images_per_folder:
                break  # Stop processing after reaching the limit
            
            file_path = os.path.join(root, file_name)
            
            # Process only image files
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                print(f"Processing image from TrueFace_PreSocial: {file_path}")
                
                try:
                    # Optionally load and display the image
                    image = mpimg.imread(file_path)
                    
                    # Copy the file to the export path
                    shutil.copy2(file_path, export_path)
                    print(f"Copied to: {os.path.join(export_path, file_name)}")
                    
                    # Increment the image counter
                    image_count += 1
                except Exception as e:
                    print(f"Error processing image {file_path}: {e}")
    else:
        # If not TrueFace_PreSocial, copy only the folder structure (no files)
        print(f"Skipping file copy for non-TrueFace_PreSocial directory: {root}")







######################## TO COPY ALLLLLLLLLLLL

# import os
# import shutil
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# ### export images with same structure

# path = "/media/NAS/TrueFace/TrueFace"
# export_mainpath = "/media/SSD_mmlab/martina.dangelo/fake-image-detectors/dataset/media/NAS/TrueFace/TrueFace"
# max_images_per_folder = 100  # Maximum number of images to copy per subfolder

# # Walk through the directory structure
# for root, dirs, files in os.walk(path):
#     # Determine the relative path from the root directory
#     rel_path = os.path.relpath(root, path)
    
#     # Create the equivalent path in the export directory
#     export_path = os.path.join(export_mainpath, rel_path)
#     os.makedirs(export_path, exist_ok=True)
    
#     image_count = 0  # Initialize image counter for the current subfolder
    
#     for file_name in files:
#         if image_count >= max_images_per_folder:
#             break  # Stop processing after 100 images
        
#         file_path = os.path.join(root, file_name)
        
#         # Process only image files
#         if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
#             print(f"Processing image: {file_path}")
            
#             # Load and display the image (optional for visualization)
#             try:
#                 image = mpimg.imread(file_path)
#                 # Copy the file to the export path
#                 shutil.copy2(file_path, export_path)
#                 print(f"Copied to: {os.path.join(export_path, file_name)}")
                
#                 # Increment the image counter
#                 image_count += 1
#             except Exception as e:
#                 print(f"Error processing image {file_path}: {e}")


