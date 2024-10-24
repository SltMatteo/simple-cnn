import os
import random
import shutil

def move_random_images(src_folder, dst_folder, num_images=50):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for class_name in os.listdir(src_folder):
        class_folder = os.path.join(src_folder, class_name)
        
        if os.path.isdir(class_folder):
            images = [f for f in os.listdir(class_folder) if f.endswith('.jpg')]
            
            selected_images = random.sample(images, min(num_images, len(images)))

            for image_name in selected_images:
                src_image_path = os.path.join(class_folder, image_name)
                
                new_image_name = f"{class_name}_{image_name}"
                dst_image_path = os.path.join(dst_folder, new_image_name)

                shutil.move(src_image_path, dst_image_path)
                print(f"Moved {src_image_path} to {dst_image_path}")

# Example usage:
src_folder = 'images/test'  # Folder containing class subfolders
dst_folder = 'mixed'  # Folder where images will be moved
move_random_images(src_folder, dst_folder, num_images=50)
