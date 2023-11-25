# scCrab: a reference-guided cancer cell identification method based on Bayesian neural networks

## Quick Start

### Input:

**h5ad file list**: File list object that includes the training set (need to be annotated), reference set and test set.

### Output:

**pred**: Array object that contains the predicted label.

**prob**: Array object that contains the predicted probabilities.

### Using tutorial:

#### Preparation
Please ensure that all the packages in the ``requirements.txt`` file have been correctly downloaded and installed. Simultaneously, create a new folder named ``model_pth`` at the top level directory (i.e., one level above the folder where the original code resides). Place all the ipynb files to be executed in the folder that is one level above the original code folder.

#### Running Process
First, construct a file list that includes the training set, reference set, and test set. This can be done using either the provided dataset or the user's own dataset. Please note that each of the AnnData objects here have undergone preprocessing. Specifically, the count matrix has been normalized to the total number of reads per cell and log transformed with a base of 2 and a pseudocount of 1. 

```python
from pathlib import Path
import anndata
import torch
names = ['BI','KLA','KLB','LA','MA','TIR','TIS']
adatas = {}
for name in names:
    adatas[name] = anndata.read_h5ad(file_path+name+'.h5ad')
```

Then, verify if the GPU is available for use.

```python
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
```

Finally, utilize the ``scCrab`` function to obtain the predicted labels and probabilities for cancer cells. Please ensure that the training set includes labels for cancer cells as well as the subcategory labels for cell types.

```python
from scCrab import *
pred, prob = scCrab(adatas=adatas, train_name="BI",reference_name="TIR",test_name="MA",device="cuda:0",
       main_obs='major_ct', detail_obs='sub_ct', save_path=result_path)
```

Additionally, the ``evaluate_metrics`` function is ready for use to obtain AUPRC, BAS, and kappa values of the prediction.

```python
AUPRC, bas, kappa = evaluate_metrics(adatas, test_name="MA", main_obs="major_ct", pred=pred, prob=prob)
```

The source datasets are available at [here](https://www.dropbox.com/scl/fi/55hhr8izbpndgha1d9eju/datasets.zip?rlkey=9ugqimbe5u4swls3fitru3k7p&dl=0).
