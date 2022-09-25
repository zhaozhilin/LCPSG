# LCPSG
Lack Content Of Punctuation Sentence Generation

This paper mainly uses the pre training language model BART of text generation to generate the missing shared components of input text under the premise of removing the assumption of conditional independence of the first and last positions of missing components; The type prediction multi task learning mechanism of sharing mode is added to improve the model's ability to learn the grammar rules of clause complex, so as to improve the model's performance; In the process of hierarchical header structure analysis, in addition to learning and forecasting the direct headers, it can also learn and predict the indirect headers nested in the hierarchical structure.

# Version
pytorch1.7
transformer4.3


# Baselines for LCPSG


## Usage

### Step 1: Install Dependency Requirements

This baseline used `PyTorch>=1.7.0`.

### Step 2: Training

```shell
python -u GenBART.py
```

### Step 3: Evaluation

**TEST**:

```shell
python predictor.py
```

## Baseline Results

<table>
  <tr>
    <td colspan=9 align="center"> epoch=12, batch=16 </td>
  </tr>
  <tr>
    <td></td>
    <td colspan=4 align="center">EVAL</td>
    <td colspan=4 align="center">TEST</td>
  </tr>
    <td align="center"></td>
    <td align="center">S</td>
    <td align="center">P</td>
    <td align="center">R</td>
    <td align="center">F1</td>
    <td align="center">S</td>
    <td align="center">P</td>
    <td align="center">R</td>
    <td align="center">F1</td>
  </tr>
  </tr>
    <td align="left">BERT</td>
    <td align="center">0.8831</td>
    <td align="center">0.6605</td>
    <td align="center">0.6762</td>
    <td align="center">0.6682</td>
    <td align="center">0.9042</td>
    <td align="center">0.7354</td>
    <td align="center">0.7429</td>
    <td align="center">0.7391</td>
  </tr>
</table>


