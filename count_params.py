import torch
from src.supernet import TaskAwareSupernet
from src.ops import OP_NAMES

architectures = {
    0: ['ResidualBN', 'MBConv 5x5', 'ResidualBN', 'MBConv 5x5', 'MBConv 5x5', 'AvgPool 3x3', 'AvgPool 3x3', 'AvgPool 3x3'],
    1: ['MBConv-SE', 'MBConv 3x3', 'SepConv 5x5', 'MBConv 3x3', 'MBConv 5x5', 'MBConv 5x5', 'MBConv 5x5', 'MBConv 5x5'],
    2: ['AvgPool 3x3', 'AvgPool 3x3', 'AvgPool 3x3', 'MBConv 3x3', 'MBConv 3x3', 'MBConv 5x5', 'MBConv 5x5', 'MBConv 5x5']
}

# The default architecture used in MT-DARTS experiments has 8 layers, channels=128, img_size=64
model = TaskAwareSupernet(num_tasks=3, num_layers=8, channels=128, img_size=64)

for task_id in range(3):
    arch = architectures[task_id]
    for l, op in enumerate(arch):
        op_mapped = op.replace(' ', '').replace('-', '')
        op_idx = OP_NAMES.index(op_mapped)
        model.alphas.data[task_id, l, :] = -100.0
        model.alphas.data[task_id, l, op_idx] = 100.0

    discrete = model.discretize(task_id)
    print(f"Task {task_id} parameters: {sum(p.numel() for p in discrete.parameters()):,}")
