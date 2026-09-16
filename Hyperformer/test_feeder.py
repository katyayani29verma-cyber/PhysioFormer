# test_feeder.py

from feeders.feeder_ntu import Feeder

dataset = Feeder(
    data_path="../physio_dataset.npz",
    split="train",
    window_size=64,
    p_interval=[1]
)

print("Dataset length:", len(dataset))

x, y, idx = dataset[0]

print("x shape:", x.shape)
print("label:", y)
print("index:", idx)