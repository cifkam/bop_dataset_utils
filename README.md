Subset of code from the [HappyPose](https://github.com/agimus-project/happypose) repository. Contains code for loading BOP datasets. Work in progress.

Example usage:
```
from bop_dataset_utils import BOPDataset

ds_name = 'ycbv'
split = 'test'

ds_dir = Path('path_to_bop_datasets')/ds_name
ds = BOPDataset(ds_dir=ds_dir, label_format=ds_name+"-{label}", split=split, load_depth=True)
```
(The rest of the code is not tested.)

### Installation
```
pip install git+https://github.com/cifkam/bop_dataset_utils.git
```
