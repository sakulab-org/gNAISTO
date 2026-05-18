# gNAISTO - Estimation of  gene regulation extending a gene network with expression data.

gNAISTO (graphical Network Assembly and Inference using Stepwise Testing with Optimization) is a Python package for inferring novel regulatory relationships connected to a known gene regulatory network using gene expression data.

The method incorporates prior network information and evaluates each candidate gene individually to support network expansion. More details about the method and its performance can be found in the corresponding paper:

```
Paper information will be added here.
```

## Installation

Clone this repository:

```
git clone git@github.com:sakulab-org/gNAISTO.git
cd gNAISTO
```

Install the package:

```
pip install -e .
```

## Usage

Example:

```python
import numpy as np
from gnaisto import estimate_regulation, prepare_data

fn_expression = "example_data/expression.csv"
fn_regulation = "example_data/regulation.csv"

expression, genename, samplename, regulation = prepare_data(
    fn_expression,
    fn_regulation
)

result_matrix, result_table = estimate_regulation(expression, regulation)
```

## Input files

### Expression data

A CSV file containing gene expression values.

Example:

```text
gene,sample1,sample2,sample3
geneA,0.1,0.3,0.2
geneB,1.2,1.0,1.4
geneC,0.5,0.4,0.7
geneD,0.8,0.9,0.6
geneE,0.2,0.1,0.3
```

### Regulation data

A CSV file containing known regulatory relationships.

The regulation matrix should represent activation (1), repression (-1), or absence of regulation (0).
Contradicted relationships (e.g., geneA activates geneB while geneB represses geneA) are ignored.

Example:

```text
,geneA,geneB,geneC
geneA,0,1,0
geneB,1,0,-1
geneC,0,-1,0
```

## Example data

Example input files are provided in:

```text
examples/example_data/
```

## Repository structure

```text
gnaisto/
├── gnaisto/              # Source code
├── examples/example_data/         # Example input data
├── environment.yml       # Conda environment file
├── README.md
└── LICENSE
```

## Paper replication

The code for replicating the results in the paper is available in [the gNAISTO-paper repository](https://github.com/sakulab-org/gNAISTO-paper).

```

## License

This project is licensed under the MIT License.

## Citation

If you use this software, please cite the corresponding paper:

```text
Citation information will be added here.
```