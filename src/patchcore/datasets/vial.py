import os
from enum import Enum

import PIL
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test_public"


class VialDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for Vial dataset (similar to MVTec AD structure).
    """

    def __init__(
        self,
        source,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the Vial data folder.
            resize: [int]. (Square) Size the loaded image initially gets resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.train_val_split = train_val_split

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        # For large images, we may want to skip cropping to preserve full size for tiling
        # But for consistency with training, we still do resize and crop
        # In practice, you might want to load full-size images when using tiling
        if imagesize >= resize:
            # If imagesize >= resize, skip cropping (just resize)
            self.transform_img = [
                transforms.Resize((imagesize, imagesize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
            self.transform_mask = [
                transforms.Resize((imagesize, imagesize)),
                transforms.ToTensor(),
            ]
        else:
            self.transform_img = [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
            self.transform_mask = [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
            ]
        
        self.transform_img = transforms.Compose(self.transform_img)
        self.transform_mask = transforms.Compose(self.transform_mask)

        self.imagesize = (3, imagesize, imagesize)
        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split(os.sep)[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        classname = "vial"  # Single class dataset

        if self.split == DatasetSplit.TRAIN:
            # Training data: only good samples
            train_path = os.path.join(self.source, "train", "good")
            if os.path.exists(train_path):
                train_files = sorted([f for f in os.listdir(train_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                imgpaths_per_class[classname] = {
                    "good": [os.path.join(train_path, f) for f in train_files]
                }
        elif self.split == DatasetSplit.VAL:
            # Validation data: only good samples
            val_path = os.path.join(self.source, "validation", "good")
            if os.path.exists(val_path):
                val_files = sorted([f for f in os.listdir(val_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                imgpaths_per_class[classname] = {
                    "good": [os.path.join(val_path, f) for f in val_files]
                }
        elif self.split == DatasetSplit.TEST:
            # Test data: good and bad samples
            test_good_path = os.path.join(self.source, "test_public", "good")
            test_bad_path = os.path.join(self.source, "test_public", "bad")
            ground_truth_path = os.path.join(self.source, "test_public", "ground_truth", "bad")

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            # Good samples
            if os.path.exists(test_good_path):
                good_files = sorted([f for f in os.listdir(test_good_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                imgpaths_per_class[classname]["good"] = [
                    os.path.join(test_good_path, f) for f in good_files
                ]
                maskpaths_per_class[classname]["good"] = [None] * len(good_files)

            # Bad samples with masks
            if os.path.exists(test_bad_path):
                bad_files = sorted([f for f in os.listdir(test_bad_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                imgpaths_per_class[classname]["bad"] = [
                    os.path.join(test_bad_path, f) for f in bad_files
                ]
                if os.path.exists(ground_truth_path):
                    mask_files = sorted([f for f in os.listdir(ground_truth_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                    maskpaths_per_class[classname]["bad"] = [
                        os.path.join(ground_truth_path, f) for f in mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["bad"] = [None] * len(bad_files)

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        if classname in maskpaths_per_class and anomaly in maskpaths_per_class[classname]:
                            data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                        else:
                            data_tuple.append(None)
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate

