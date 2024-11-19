import os
import random

# image_folder = '/data1/phuc/weizmann-horse-database/horse'
# mask_folder = '/data1/phuc/weizmann-horse-database/mask'

image_folder = '/data1/phuc/kitti/data_road/training/image_2'
mask_folder = '/data1/phuc/kitti/data_road/training/gt_image_2'

images = sorted([img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])
random.shuffle(images)

split_index = int(0.7 * len(images))
train_images = images[:split_index]
test_images = images[split_index:]

folder = "kitti"
with open(os.path.join(folder, 'train.txt'), 'w') as train_file, open(os.path.join(folder, 'test.txt'), 'w') as test_file:
    # for img in train_images:
    #     train_file.write(f"{os.path.join(image_folder, img)},{os.path.join(mask_folder, img)}\n")
    # for img in test_images:
    #     test_file.write(f"{os.path.join(image_folder, img)},{os.path.join(mask_folder, img)}\n")

    for img in train_images:
        tmp = img.split("_")
        mask_name = tmp[0] + "_road_" + tmp[1]
        train_file.write(f"{os.path.join(image_folder, img)},{os.path.join(mask_folder, mask_name)}\n")
    for img in test_images:
        tmp = img.split("_")
        mask_name = tmp[0] + "_road_" + tmp[1]
        test_file.write(f"{os.path.join(image_folder, img)},{os.path.join(mask_folder, mask_name)}\n")

print("train.txt and test.txt files created successfully.")
