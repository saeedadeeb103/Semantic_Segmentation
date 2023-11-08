import os
import shutil

class DatasetOrganizer:
    """
    A class for organizing a dataset into 'images' and 'masks' folders.
    """
    def __init__(self, dataset_dir):
        """
        Initialize the DatasetOrganizer with the dataset directory.

        Args:
            dataset_dir (str): The path to the root dataset directory.
        """
        self.dataset_dir = dataset_dir

    def create_subdirectories(self, folder):
        """
        Create 'images' and 'masks' subdirectories within the specified folder.

        Args:
            folder (str): The path to the folder where subdirectories will be created.
        """
        image_dir = os.path.join(folder, "images")
        mask_dir = os.path.join(folder, "masks")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

    def organize_dataset(self, folder_name):
        """
        Organize the images and masks within a specific folder.

        Args:
            folder_name (str): Name of the folder ('train' or 'val') to be organized.
        """
        folder_dir = os.path.join(self.dataset_dir, folder_name)
        self.create_subdirectories(folder_dir)

        for file in os.listdir(folder_dir):
            if file.endswith(".jpg"):
                image_src = os.path.join(folder_dir, file)
                image_dst = os.path.join(folder_dir, "images", file)
                shutil.move(image_src, image_dst)
            elif file.endswith("_mask.png"):
                mask_src = os.path.join(folder_dir, file)
                mask_dst = os.path.join(folder_dir, "masks", file)
                shutil.move(mask_src, mask_dst)

        print(f"Organized the '{folder_name}' dataset into 'images' and 'masks' folders.")

if __name__ == "__main__":
    # Specify the root dataset directory.
    dataset_dir = "dataset"

    # Create a DatasetOrganizer instance.
    organizer = DatasetOrganizer(dataset_dir)

    # Organize the 'train' and 'val' folders.
    organizer.organize_dataset("train")
    organizer.organize_dataset("valid")
