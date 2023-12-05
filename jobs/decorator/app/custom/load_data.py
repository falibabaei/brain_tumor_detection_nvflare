import os
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import json


class TumorImageDataset(Dataset):
    """load, transform and return image and label"""

    def __init__(self, annotations_df, img_dir, transform=None):
        self.img_labels = annotations_df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # get image path according to idx
        img_path = os.path.join(
            self.img_dir, self.img_labels.iloc[idx, 0]
        )
        # convert all image to RGB format
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]
        # apply image transform
        if self.transform:
            image = self.transform(image)
        return [image, label]


def create_dataset(
    data_split, img_dir, img_list, index, image_transform
):
    start, end = data_split[index]["start"], data_split[index]["end"]
    images = img_list[start:end]
    img_names, img_labels = zip(
        *[(i, 0 if "Not Cancer" in i else 1) for i in images]
    )
    names = pd.Series(img_names, name="name")
    labels = pd.Series(img_labels, name="label")
    dataset_df = pd.concat([names, labels], axis=1)
    return TumorImageDataset(dataset_df, img_dir, image_transform)


def load_data(
    data_split: dict, client_id: str, image_transform, batch_size: int
):
    """
    Load and prepare training and validation data from a data split file (json) for a specific client.

    Parameters:
    - data_split_filename (str): The file path to the data split JSON file containing information about the data partition.
    - client_id (str): The identifier of the client for which the data is loaded.
    - image_transform: A function to transform images used for data augmentation.

    Returns:
    Tuple[DataLoader, DataLoader]: A tuple containing training and validation data loaders.

    Raises:
    - ValueError: If the specified client_id is not found in the data split or if the data split
      lacks a validation split.
    """

    # with open(data_split_filename, "r") as file:
    #    data_split = json.load(file)

    data_index = data_split["data_index"]

    if client_id not in data_index.keys():
        raise ValueError(
            f"Data does not contain Client {client_id} split",
        )

    if "valid" not in data_index.keys():
        raise ValueError(
            "Data does not contain Validation split",
        )

    img_dir = data_split["data_path"]
    img_list = os.listdir(img_dir)
    random.shuffle(img_list)
    train_data = create_dataset(
        data_split["data_index"],
        img_dir,
        img_list,
        client_id,
        image_transform,
    )
    valid_data = create_dataset(
        data_split["data_index"],
        img_dir,
        img_list,
        "valid",
        image_transform,
    )

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=False
    )
    valid_dataloader = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False
    )

    return train_data, train_dataloader, valid_data, valid_dataloader


if __name__ == "__main__":
    image_transform = transforms.Compose(
        [
            transforms.Resize(size=(256, 256)),
            transforms.CenterCrop(size=(244, 244)),
            transforms.ToTensor(),
        ]
    )
    client_id = "site1"
    data_split_filename = "/home/se1131/brain_scan/Brain_Tumor_DataSet/Brain_Tumor_DataSet/data_split.json"
    with open(data_split_filename, "r") as file:
        data_split = json.load(file)
    train_data, vaid_data = load_data(
        data_split, client_id, image_transform
    )
