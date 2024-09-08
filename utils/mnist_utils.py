import numpy as np


class MNISTUtils:
    @staticmethod
    def read_mnist_images(file_path):
        with open(file_path, "rb") as f:
            magic_number = int.from_bytes(f.read(4), "big")
            num_images = int.from_bytes(f.read(4), "big")
            num_rows = int.from_bytes(f.read(4), "big")
            num_cols = int.from_bytes(f.read(4), "big")

            image_data = np.frombuffer(f.read(), dtype=np.uint8)

        # Reshape to (num_images, num_rows, num_cols)
        images = image_data.reshape(num_images, num_rows, num_cols)

        # Normalize
        images = images / 255.0

        # Flatten the images (called feature matrix)
        flattened_images = images.reshape(num_images, -1)

        return flattened_images

    @staticmethod
    def read_mnist_labels(file_path):
        with open(file_path, "rb") as f:
            magic_number = int.from_bytes(f.read(4), "big")
            num_labels = int.from_bytes(f.read(4), "big")

            label_data = np.frombuffer(f.read(), dtype=np.uint8)

        return label_data
