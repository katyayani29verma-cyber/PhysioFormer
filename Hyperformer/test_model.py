import torch

from feeders.feeder_ntu import Feeder
from model.Hyperformer import Model


dataset = Feeder(
    data_path="../physio_dataset.npz",
    split="train",
    window_size=64,
    p_interval=[1]
)

x, y, idx = dataset[0]

print("Input shape:", x.shape)

x = torch.tensor(x).float()

x = x.unsqueeze(0)

print("Batch shape:", x.shape)

model = Model(
    num_class=2,
    num_point=18,
    num_person=1,
    graph="graph.coco18.Graph",
    graph_args={},

    joint_label=[
        0,0,
        1,1,1,1,1,1,
        2,2,2,2,2,2,
        3,3,3,3
    ]
)

output, _ = model(x, None)

print("Output shape:", output.shape)
print("Output tensor:")
print(output)
import torch.nn.functional as F

output, _ = model(x, None)

prob = F.softmax(output, dim=1)

print("Probabilities:")
print(prob)

confidence, pred = torch.max(prob, dim=1)

print("Predicted Class:", pred.item())
print("Confidence:", confidence.item())