from pathlib import Path

from inspector.seeder import seed_everything

from inspector.data.pems_dataset import PEMS
from inspector.data.data_utils import PEMSTransform

if __name__ == "__main__":
    seed_everything(1111)
    p = Path("./pems")
    p.mkdir(exist_ok=True, parents=True)

    _ = PEMS(p, pre_transform=PEMSTransform(), force_reload=True)
