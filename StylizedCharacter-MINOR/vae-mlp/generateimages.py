import os
import shutil

def rename_images(source_folder, destination_folder):
    files = os.listdir(source_folder)
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for i, filename in enumerate(files[:50], start=1):
        _, ext = os.path.splitext(filename)
        new_filename = str(i) + ext
        shutil.copyfile(os.path.join(source_folder, filename), os.path.join(destination_folder, new_filename))

def rename_and_copy_images(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    files = os.listdir(source_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    images = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]

    for index, image in enumerate(images, start=1):
        file_extension = os.path.splitext(image)[1]
        new_file_name = f"{index}{file_extension}"
        source_path = os.path.join(source_folder, image)
        destination_path = os.path.join(destination_folder, new_file_name)
        shutil.copy2(source_path, destination_path)

    print(f"Renamed and copied {len(images)} images to {destination_folder}")

source_folder = 'data/human_expression/surprise'
destination_folder = 'data5-tomnjerry/human_expression/surprise'
rename_and_copy_images(source_folder, destination_folder)

# rename_images(source_folder, destination_folder)

