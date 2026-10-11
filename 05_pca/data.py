import os
import numpy as np
from PIL import Image

IMAGE_SIZE = 20


def make_toy_data(n: int = 300, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)

    x = rng.normal(0, 3.0, size = (n, 1))
    y = rng.normal(0, 0.5, size = (n, 1))
    points = np.hstack([x,y])

    angle = np.pi/6
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])

    rotation_points = points @ rotation_matrix.T

    shift = np.array([2.0, 4.0])
    final_points = rotation_points + shift

    return final_points


def load_fruit_vectors(split: str = "train", data_dir: str = "../04_fruit_network/data") -> tuple:
    split_dir = os.path.join(data_dir, split)

    class_names = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir,d))])

    X_list = []
    labels_list = []

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(split_dir, class_name)
        image_filenames = sorted(os.listdir(class_dir))

        for img_name in image_filenames:
            img_path = os.path.join(class_dir, img_name)
            if not os.path.isfile(img_path):
                continue

            with Image.open(img_path) as img:
                img = img.convert("L")
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

                img_array = np.asarray(img, dtype = np.float32) / 255.0
                flat_vector = img_array.flatten()

                X_list.append(flat_vector)
                labels_list.append(label_idx)

    X = np.stack(X_list, axis = 0) if X_list else np.empty((0, IMAGE_SIZE * IMAGE_SIZE), dtype = np.float32)
    labels = np.array(labels_list, dtype = np.int64)

    return (X, labels, class_names)


if __name__ == "__main__":
    toy = make_toy_data()
    print(toy.shape)
    X, labels, classes = load_fruit_vectors()
    print(X.shape, labels.shape, classes)