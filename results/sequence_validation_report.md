# B3 — SEQUENCE VALIDATION REPORT

## Overview

Sequence builder successfully constructed 72-hour historical input ($X$) to 72-hour future target ($y$) sequence windows for PM2.5 forecasting.

## Sequence Shapes and Statistics

| Dataset Split | X Shape (Input) | y Shape (Target) | Number of Samples | Memory (MB) |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | `(19005, 72, 49)` | `(19005, 72, 1)` | 19005 | 255.77 MB |
| **Validation** | `(4493, 72, 49)` | `(4493, 72, 1)` | 4493 | 60.47 MB |
| **Test** | `(3393, 72, 49)` | `(3393, 72, 1)` | 3393 | 45.66 MB |
| **Total** | - | - | **26891** | **361.91 MB** |

## Programmatic Integrity Checks (10 Rules)

| # | Verification Rule | Result | Details |
| :---: | :--- | :---: | :--- |
| 1 | X has exactly 72 timesteps | **PASS** | Verified programmatically |
| 2 | y has exactly 72 timesteps | **PASS** | Verified programmatically |
| 3 | X has exactly 49 features | **PASS** | Verified programmatically |
| 4 | y has exactly 1 target | **PASS** | Verified programmatically |
| 5 | target_start = input_end + 1 hour | **PASS** | Verified programmatically |
| 6 | Continuous hourly timestamps inside every sequence | **PASS** | Verified programmatically |
| 7 | No sequence crosses dataset split boundaries | **PASS** | Verified programmatically |
| 8 | No future target leakage in X | **PASS** | Verified programmatically |
| 9 | PM2.5 target contains no NaN | **PASS** | Verified programmatically |
| 10 | No duplicate sequences generated | **PASS** | Verified programmatically |

---

## Concrete Timestamp Sequence Examples

### Train Partition Examples

| Sample Index | Input Start ($t-71$) | Input End ($t$) | Target Start ($t+1$) | Target End ($t+72$) | $\Delta(t_{\text{target, start}} - t_{\text{input, end}})$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Sample #0 | `2015-05-01 09:00:00` | `2015-05-04 08:00:00` | `2015-05-04 09:00:00` | `2015-05-07 08:00:00` | `1 hour (EXACT)` |
| Sample #9502 | `2018-10-13 13:00:00` | `2018-10-16 12:00:00` | `2018-10-16 13:00:00` | `2018-10-19 12:00:00` | `1 hour (EXACT)` |
| Sample #19004 | `2021-12-20 21:00:00` | `2021-12-23 20:00:00` | `2021-12-23 21:00:00` | `2021-12-26 20:00:00` | `1 hour (EXACT)` |


### Validation Partition Examples

| Sample Index | Input Start ($t-71$) | Input End ($t$) | Target Start ($t+1$) | Target End ($t+72$) | $\Delta(t_{\text{target, start}} - t_{\text{input, end}})$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Sample #0 | `2022-01-01 00:00:00` | `2022-01-03 23:00:00` | `2022-01-04 00:00:00` | `2022-01-06 23:00:00` | `1 hour (EXACT)` |
| Sample #2246 | `2022-08-19 17:00:00` | `2022-08-22 16:00:00` | `2022-08-22 17:00:00` | `2022-08-25 16:00:00` | `1 hour (EXACT)` |
| Sample #4492 | `2022-12-26 00:00:00` | `2022-12-28 23:00:00` | `2022-12-29 00:00:00` | `2022-12-31 23:00:00` | `1 hour (EXACT)` |


### Test Partition Examples

| Sample Index | Input Start ($t-71$) | Input End ($t$) | Target Start ($t+1$) | Target End ($t+72$) | $\Delta(t_{\text{target, start}} - t_{\text{input, end}})$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Sample #0 | `2023-01-01 00:00:00` | `2023-01-03 23:00:00` | `2023-01-04 00:00:00` | `2023-01-06 23:00:00` | `1 hour (EXACT)` |
| Sample #1696 | `2023-03-24 19:00:00` | `2023-03-27 18:00:00` | `2023-03-27 19:00:00` | `2023-03-30 18:00:00` | `1 hour (EXACT)` |
| Sample #3392 | `2023-12-26 00:00:00` | `2023-12-28 23:00:00` | `2023-12-29 00:00:00` | `2023-12-31 23:00:00` | `1 hour (EXACT)` |


## Final B3 Status

```
READY FOR B4 — LEAKAGE-SAFE SCALING
```
