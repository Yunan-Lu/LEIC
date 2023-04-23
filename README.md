# LEIC
LEIC: a **L**abel **E**nhancement model via joint **I**mplicit representation **C**lustering

## Environments
python=3.8.16, numpy=1.23.5, scipy=1.10.0, pytorch=2.0.0+cpu, scikit-learn=1.2.2.

## Reproducing
Change the directory to this project and run the following command in terminal.
```Terminal
python demo.py
```


## Usage
Here is a simple example of using LEIC.
```python
import numpy as np
from leic import LEIC
from utils import report, binarize

# load data
X, D = load_dataset('SBU-3DFE') # this api should be defined by users
L = binarize(D)

# train without early-stop trick
model = LEIC(trace_step=np.inf).fit(X, L)
# show the recovery performance
Drec = model.label_distribution_
report(Drec, D)

# train with early-stop trick
model = LEIC(trace_step=20).fit(X, L)
Drec_trace = model.trace_
# show the recovery performance
for k in Drec_trace.keys():
    print("The recovery performance at %d-th iteration:" % k)
    report(Drec_trace[k], D)
```

## Datasets
- The datasets used in our work is partially provided by [PALM](http://palm.seu.edu.cn/xgeng/LDL/index.htm)
- Emotion6: [http://chenlab.ece.cornell.edu/people/kuanchuan/index.html](http://chenlab.ece.cornell.edu/people/kuanchuan/index.html)
- Twitter-LDL: [http://47.105.62.179:8081/sentiment/index.html](http://47.105.62.179:8081/sentiment/index.html)

## Paper
```latex
@inproceedings{Lu2023LEIC,
	title={Label Enhancement via Joint Implicit Representation Clustering},
	author={Yunan Lu and Weiwei Li and Xiuyi Jia},
	booktitle={International Joint Conferences on Artificial Intelligence},
	year={2023},
}
```
