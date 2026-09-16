from __future__ import annotations

import numpy as np

from torch.utils.data import Dataset


class Feeder(Dataset):

    def __init__(
        self,
        data_path,
        label_path=None,
        p_interval=(1,),
        split="train",
        random_choose=False,
        random_shift=False,
        random_move=False,
        random_rot=False,
        window_size=64,
        normalization=False,
        debug=False,
        use_mmap=False,
        bone=False,
        vel=False,
    ):

        self.data_path = data_path
        self.split = split
        self.debug = debug
        self.archive = None

        if random_rot:
            raise ValueError(
                "random_rot must be False for "
                "x,y,vx,vy data."
            )

        if bone:
            raise ValueError(
                "bone=True is not supported."
            )

        if vel:
            raise ValueError(
                "vel=True must not be used because "
                "vx and vy already exist."
            )

        self.load_data()

    def load_data(self):

        self.archive = np.load(
            self.data_path,
        )

        if self.split == "train":

            self.data = (
                self.archive["x_train"]
            )

            labels = (
                self.archive["y_train"]
            )

            names_key = "train_names"

        elif self.split == "test":

            self.data = (
                self.archive["x_test"]
            )

            labels = (
                self.archive["y_test"]
            )

            names_key = "test_names"

        else:

            raise ValueError(
                "split must be train or test"
            )

        if (
            self.data.ndim != 5
            or
            self.data.shape[1:]
            != (4, 64, 18, 1)
        ):

            raise ValueError(
                f"Wrong data shape: "
                f"{self.data.shape}"
            )

        if (
            labels.ndim != 2
            or
            labels.shape[1] != 49
        ):

            raise ValueError(
                f"Wrong label shape: "
                f"{labels.shape}"
            )

        self.label = labels.argmax(
            axis=1
        ).astype(np.int64)

        if names_key in self.archive:

            self.sample_name = (
                self.archive[names_key]
                .tolist()
            )

        else:

            self.sample_name = [
                f"{self.split}_{index}"
                for index in range(
                    len(self.data)
                )
            ]

        if self.debug:

            limit = min(
                100,
                len(self.data),
            )

            self.data = self.data[:limit]
            self.label = self.label[:limit]
            self.sample_name = (
                self.sample_name[:limit]
            )

        print(
            f"Loaded {self.split}: "
            f"{self.data.shape}"
        )

    def __len__(self):

        return len(
            self.label
        )

    def __getitem__(self, index):

        data_numpy = np.asarray(
            self.data[index],
            dtype=np.float32,
        ).copy()

        label = int(
            self.label[index]
        )

        return (
            data_numpy,
            label,
            index,
        )

    def top_k(
        self,
        score,
        top_k,
    ):

        rank = score.argsort()

        hits = [
            label in rank[index, -top_k:]
            for index, label
            in enumerate(self.label)
        ]

        return (
            sum(hits)
            / len(hits)
        )