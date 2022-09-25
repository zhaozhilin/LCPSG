# LCPSG
Lack Content Of Punctuation Sentence Generation

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


