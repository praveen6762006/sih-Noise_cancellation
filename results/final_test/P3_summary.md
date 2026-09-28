# Priority 3: target verification, deployed path vs PyTorch

Model H. **Deployed path** = NumPy front end + reference guard + streaming ONNX Runtime + ~1.5 s automatic level control (what runs on the target). **PyTorch path** = per-clip RMS normalisation, offline STFT. The two are never averaged together; they are compared clip by clip.

Target verdicts per class x input-SNR cell (25 clips): PASS = the 95% bootstrap interval of the cell mean is entirely above the threshold; FAIL = entirely below; MARG = the interval contains the threshold. Two readings of the SNR target are shown because the statement says only "SNR > 15 dB": **dSNR** = SI-SNR improvement > 15 dB (strict, primary), **oSNR** = output SI-SNR > 15 dB (looser, used earlier in this project). SI-SNR is measured against the dereverberated direct-path target.

## final_A (unused test-clean utterances)

### final_A: DEPLOYED (ONNX fp32) full target matrix

| class | input SNR | n | dSI-SNR | dSNR>15 | out SI-SNR | oSNR>15 | STOI | >0.85 | PESQ | >2.5 | **ALL (strict, dSNR)** | ALL (alt, oSNR) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| engine | -10 | 25 | +15.4 | marg | +5.3 | fail | 0.778 | fail | 1.39 | fail | **fail** | fail |
| engine | -5 | 25 | +11.9 | fail | +6.9 | fail | 0.821 | marg | 1.49 | fail | **fail** | fail |
| engine | +0 | 25 | +12.2 | fail | +12.2 | fail | 0.925 | PASS | 2.13 | fail | **fail** | fail |
| engine | +5 | 25 | +8.6 | fail | +13.5 | fail | 0.943 | PASS | 2.39 | marg | **fail** | fail |
| engine | +10 | 25 | +7.7 | fail | +17.4 | PASS | 0.968 | PASS | 2.94 | PASS | **fail** | PASS |
| engine | +15 | 25 | +5.0 | fail | +19.3 | PASS | 0.973 | PASS | 3.24 | PASS | **fail** | PASS |
| helicopter | -10 | 25 | +13.4 | fail | +3.3 | fail | 0.676 | fail | 1.19 | fail | **fail** | fail |
| helicopter | -5 | 25 | +10.4 | fail | +5.4 | fail | 0.786 | fail | 1.29 | fail | **fail** | fail |
| helicopter | +0 | 25 | +9.7 | fail | +9.7 | fail | 0.875 | PASS | 1.63 | fail | **fail** | fail |
| helicopter | +5 | 25 | +7.4 | fail | +12.3 | fail | 0.914 | PASS | 2.02 | fail | **fail** | fail |
| helicopter | +10 | 25 | +6.0 | fail | +15.8 | PASS | 0.958 | PASS | 2.61 | marg | **fail** | marg |
| helicopter | +15 | 25 | +4.0 | fail | +18.6 | PASS | 0.976 | PASS | 3.11 | PASS | **fail** | PASS |
| drone | -10 | 25 | +10.6 | fail | +0.6 | fail | 0.664 | fail | 1.15 | fail | **fail** | fail |
| drone | -5 | 25 | +9.8 | fail | +4.7 | fail | 0.769 | fail | 1.30 | fail | **fail** | fail |
| drone | +0 | 25 | +8.3 | fail | +8.2 | fail | 0.850 | marg | 1.59 | fail | **fail** | fail |
| drone | +5 | 25 | +6.6 | fail | +11.4 | fail | 0.904 | PASS | 2.02 | fail | **fail** | fail |
| drone | +10 | 25 | +5.0 | fail | +14.9 | marg | 0.946 | PASS | 2.53 | marg | **fail** | marg |
| drone | +15 | 25 | +3.1 | fail | +17.7 | PASS | 0.968 | PASS | 2.97 | PASS | **fail** | PASS |
| wind | -10 | 25 | +8.6 | fail | -1.4 | fail | 0.485 | fail | 1.07 | fail | **fail** | fail |
| wind | -5 | 25 | +8.2 | fail | +3.2 | fail | 0.643 | fail | 1.15 | fail | **fail** | fail |
| wind | +0 | 25 | +6.9 | fail | +6.9 | fail | 0.759 | fail | 1.33 | fail | **fail** | fail |
| wind | +5 | 25 | +5.6 | fail | +10.5 | fail | 0.856 | marg | 1.60 | fail | **fail** | fail |
| wind | +10 | 25 | +5.0 | fail | +14.7 | marg | 0.916 | PASS | 2.11 | fail | **fail** | fail |
| wind | +15 | 25 | +2.8 | fail | +17.1 | PASS | 0.959 | PASS | 2.52 | marg | **fail** | marg |
| siren | -10 | 25 | +20.0 | PASS | +10.0 | fail | 0.914 | PASS | 2.18 | fail | **fail** | fail |
| siren | -5 | 25 | +16.8 | PASS | +11.8 | fail | 0.932 | PASS | 2.44 | marg | **marg** | fail |
| siren | +0 | 25 | +13.7 | fail | +13.6 | fail | 0.952 | PASS | 2.88 | PASS | **fail** | fail |
| siren | +5 | 25 | +10.8 | fail | +15.7 | marg | 0.973 | PASS | 3.27 | PASS | **fail** | marg |
| siren | +10 | 25 | +7.7 | fail | +17.6 | PASS | 0.978 | PASS | 3.45 | PASS | **fail** | PASS |
| siren | +15 | 25 | +5.7 | fail | +20.1 | PASS | 0.985 | PASS | 3.76 | PASS | **fail** | PASS |
| gunshot | -10 | 25 | +15.4 | marg | +6.5 | fail | 0.811 | fail | 1.54 | fail | **fail** | fail |
| gunshot | -5 | 25 | +14.2 | marg | +9.6 | fail | 0.868 | marg | 1.94 | fail | **fail** | fail |
| gunshot | +0 | 25 | +11.7 | fail | +11.6 | fail | 0.896 | PASS | 2.05 | fail | **fail** | fail |
| gunshot | +5 | 25 | +10.7 | fail | +15.7 | marg | 0.952 | PASS | 2.81 | PASS | **fail** | marg |
| gunshot | +10 | 25 | +7.6 | fail | +17.4 | PASS | 0.961 | PASS | 2.94 | PASS | **fail** | PASS |
| gunshot | +15 | 25 | +4.2 | fail | +18.6 | PASS | 0.975 | PASS | 3.25 | PASS | **fail** | PASS |
| artillery | -10 | 25 | +17.6 | PASS | +9.9 | fail | 0.876 | PASS | 2.00 | fail | **fail** | fail |
| artillery | -5 | 25 | +16.7 | PASS | +12.6 | fail | 0.924 | PASS | 2.40 | marg | **marg** | fail |
| artillery | +0 | 25 | +14.3 | marg | +14.4 | marg | 0.944 | PASS | 2.65 | PASS | **marg** | marg |
| artillery | +5 | 25 | +12.8 | fail | +17.7 | PASS | 0.967 | PASS | 3.12 | PASS | **fail** | PASS |
| artillery | +10 | 25 | +9.8 | fail | +19.5 | PASS | 0.982 | PASS | 3.49 | PASS | **fail** | PASS |
| artillery | +15 | 25 | +7.3 | fail | +21.7 | PASS | 0.985 | PASS | 3.64 | PASS | **fail** | PASS |
| mixed | -10 | 25 | +12.2 | fail | +2.6 | fail | 0.663 | fail | 1.25 | fail | **fail** | fail |
| mixed | -5 | 25 | +11.3 | fail | +6.3 | fail | 0.760 | fail | 1.36 | fail | **fail** | fail |
| mixed | +0 | 25 | +9.8 | fail | +9.8 | fail | 0.847 | marg | 1.68 | fail | **fail** | fail |
| mixed | +5 | 25 | +8.3 | fail | +13.2 | fail | 0.927 | PASS | 2.09 | fail | **fail** | fail |
| mixed | +10 | 25 | +5.9 | fail | +15.7 | PASS | 0.956 | PASS | 2.53 | marg | **fail** | marg |
| mixed | +15 | 25 | +4.3 | fail | +18.7 | PASS | 0.974 | PASS | 3.06 | PASS | **fail** | PASS |

Cells: 48. ALL strict (dSNR): PASS 0, MARGINAL 3, FAIL 45. ALL alt (oSNR): PASS 12, MARGINAL 7, FAIL 29.

Per input SNR (cells passing ALL, strict / alt): -10 dB: 0/0 of 8, -5 dB: 0/0 of 8, +0 dB: 0/0 of 8, +5 dB: 0/1 of 8, +10 dB: 0/4 of 8, +15 dB: 0/7 of 8

Compact grid, deployed path (letters = target PASSED: D = dSNR>15, O = oSNR>15, T = STOI>0.85, P = PESQ>2.5; '.' = not passed):

| class | -10 | -5 | +0 | +5 | +10 | +15 |
|---|---|---|---|---|---|---|
| engine | .... | .... | ..T. | ..T. | .OTP | .OTP |
| helicopter | .... | .... | ..T. | ..T. | .OT. | .OTP |
| drone | .... | .... | .... | ..T. | ..T. | .OTP |
| wind | .... | .... | .... | .... | ..T. | .OT. |
| siren | D.T. | D.T. | ..TP | ..TP | .OTP | .OTP |
| gunshot | .... | .... | ..T. | ..TP | .OTP | .OTP |
| artillery | D.T. | D.T. | ..TP | .OTP | .OTP | .OTP |
| mixed | .... | .... | .... | ..T. | .OT. | .OTP |

### final_A: PyTorch full target matrix (for comparison)

| class | input SNR | n | dSI-SNR | dSNR>15 | out SI-SNR | oSNR>15 | STOI | >0.85 | PESQ | >2.5 | **ALL (strict, dSNR)** | ALL (alt, oSNR) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| engine | -10 | 25 | +15.4 | marg | +5.3 | fail | 0.779 | fail | 1.39 | fail | **fail** | fail |
| engine | -5 | 25 | +12.0 | fail | +7.0 | fail | 0.822 | marg | 1.49 | fail | **fail** | fail |
| engine | +0 | 25 | +12.3 | fail | +12.2 | fail | 0.924 | PASS | 2.13 | fail | **fail** | fail |
| engine | +5 | 25 | +8.6 | fail | +13.5 | fail | 0.943 | PASS | 2.39 | marg | **fail** | fail |
| engine | +10 | 25 | +7.7 | fail | +17.5 | PASS | 0.968 | PASS | 2.96 | PASS | **fail** | PASS |
| engine | +15 | 25 | +5.1 | fail | +19.4 | PASS | 0.974 | PASS | 3.30 | PASS | **fail** | PASS |
| helicopter | -10 | 25 | +13.5 | fail | +3.4 | fail | 0.680 | fail | 1.19 | fail | **fail** | fail |
| helicopter | -5 | 25 | +10.5 | fail | +5.5 | fail | 0.788 | fail | 1.29 | fail | **fail** | fail |
| helicopter | +0 | 25 | +9.7 | fail | +9.8 | fail | 0.875 | PASS | 1.63 | fail | **fail** | fail |
| helicopter | +5 | 25 | +7.5 | fail | +12.4 | fail | 0.914 | PASS | 2.02 | fail | **fail** | fail |
| helicopter | +10 | 25 | +5.9 | fail | +15.8 | PASS | 0.958 | PASS | 2.63 | PASS | **fail** | PASS |
| helicopter | +15 | 25 | +4.2 | fail | +18.7 | PASS | 0.976 | PASS | 3.13 | PASS | **fail** | PASS |
| drone | -10 | 25 | +11.7 | fail | +1.7 | fail | 0.676 | fail | 1.18 | fail | **fail** | fail |
| drone | -5 | 25 | +10.3 | fail | +5.3 | fail | 0.771 | fail | 1.35 | fail | **fail** | fail |
| drone | +0 | 25 | +8.4 | fail | +8.4 | fail | 0.851 | marg | 1.60 | fail | **fail** | fail |
| drone | +5 | 25 | +6.6 | fail | +11.5 | fail | 0.906 | PASS | 2.07 | fail | **fail** | fail |
| drone | +10 | 25 | +5.2 | fail | +15.1 | marg | 0.947 | PASS | 2.62 | PASS | **fail** | marg |
| drone | +15 | 25 | +3.2 | fail | +17.7 | PASS | 0.968 | PASS | 3.00 | PASS | **fail** | PASS |
| wind | -10 | 25 | +8.7 | fail | -1.4 | fail | 0.487 | fail | 1.07 | fail | **fail** | fail |
| wind | -5 | 25 | +8.3 | fail | +3.2 | fail | 0.645 | fail | 1.15 | fail | **fail** | fail |
| wind | +0 | 25 | +7.0 | fail | +6.9 | fail | 0.761 | fail | 1.33 | fail | **fail** | fail |
| wind | +5 | 25 | +5.6 | fail | +10.5 | fail | 0.857 | marg | 1.61 | fail | **fail** | fail |
| wind | +10 | 25 | +5.0 | fail | +14.8 | marg | 0.916 | PASS | 2.10 | fail | **fail** | fail |
| wind | +15 | 25 | +2.8 | fail | +17.1 | PASS | 0.958 | PASS | 2.54 | marg | **fail** | marg |
| siren | -10 | 25 | +20.8 | PASS | +10.8 | fail | 0.916 | PASS | 2.25 | fail | **fail** | fail |
| siren | -5 | 25 | +17.1 | PASS | +12.0 | fail | 0.933 | PASS | 2.50 | marg | **marg** | fail |
| siren | +0 | 25 | +14.0 | marg | +13.9 | marg | 0.953 | PASS | 2.94 | PASS | **marg** | marg |
| siren | +5 | 25 | +10.9 | fail | +15.7 | marg | 0.973 | PASS | 3.30 | PASS | **fail** | marg |
| siren | +10 | 25 | +7.8 | fail | +17.7 | PASS | 0.978 | PASS | 3.48 | PASS | **fail** | PASS |
| siren | +15 | 25 | +5.8 | fail | +20.2 | PASS | 0.985 | PASS | 3.80 | PASS | **fail** | PASS |
| gunshot | -10 | 25 | +15.6 | marg | +6.7 | fail | 0.810 | fail | 1.57 | fail | **fail** | fail |
| gunshot | -5 | 25 | +14.3 | marg | +9.7 | fail | 0.866 | marg | 1.93 | fail | **fail** | fail |
| gunshot | +0 | 25 | +11.7 | fail | +11.6 | fail | 0.894 | PASS | 2.04 | fail | **fail** | fail |
| gunshot | +5 | 25 | +10.7 | fail | +15.7 | marg | 0.951 | PASS | 2.81 | PASS | **fail** | marg |
| gunshot | +10 | 25 | +7.7 | fail | +17.5 | PASS | 0.961 | PASS | 2.97 | PASS | **fail** | PASS |
| gunshot | +15 | 25 | +4.5 | fail | +18.9 | PASS | 0.976 | PASS | 3.31 | PASS | **fail** | PASS |
| artillery | -10 | 25 | +18.0 | PASS | +10.2 | fail | 0.878 | PASS | 1.98 | fail | **fail** | fail |
| artillery | -5 | 25 | +17.0 | PASS | +12.9 | fail | 0.922 | PASS | 2.38 | marg | **marg** | fail |
| artillery | +0 | 25 | +14.3 | marg | +14.4 | marg | 0.943 | PASS | 2.63 | marg | **marg** | marg |
| artillery | +5 | 25 | +12.7 | fail | +17.6 | PASS | 0.967 | PASS | 3.10 | PASS | **fail** | PASS |
| artillery | +10 | 25 | +9.9 | fail | +19.6 | PASS | 0.982 | PASS | 3.54 | PASS | **fail** | PASS |
| artillery | +15 | 25 | +7.4 | fail | +21.8 | PASS | 0.985 | PASS | 3.71 | PASS | **fail** | PASS |
| mixed | -10 | 25 | +12.4 | fail | +2.8 | fail | 0.664 | fail | 1.24 | fail | **fail** | fail |
| mixed | -5 | 25 | +11.4 | fail | +6.4 | fail | 0.760 | fail | 1.36 | fail | **fail** | fail |
| mixed | +0 | 25 | +9.8 | fail | +9.8 | fail | 0.846 | marg | 1.69 | fail | **fail** | fail |
| mixed | +5 | 25 | +8.3 | fail | +13.2 | fail | 0.927 | PASS | 2.10 | fail | **fail** | fail |
| mixed | +10 | 25 | +6.0 | fail | +15.8 | PASS | 0.956 | PASS | 2.57 | marg | **fail** | marg |
| mixed | +15 | 25 | +4.4 | fail | +18.8 | PASS | 0.974 | PASS | 3.10 | PASS | **fail** | PASS |

Cells: 48. ALL strict (dSNR): PASS 0, MARGINAL 4, FAIL 44. ALL alt (oSNR): PASS 13, MARGINAL 7, FAIL 28.

Per input SNR (cells passing ALL, strict / alt): -10 dB: 0/0 of 8, -5 dB: 0/0 of 8, +0 dB: 0/0 of 8, +5 dB: 0/1 of 8, +10 dB: 0/5 of 8, +15 dB: 0/7 of 8

Compact grid, PyTorch path:

| class | -10 | -5 | +0 | +5 | +10 | +15 |
|---|---|---|---|---|---|---|
| engine | .... | .... | ..T. | ..T. | .OTP | .OTP |
| helicopter | .... | .... | ..T. | ..T. | .OTP | .OTP |
| drone | .... | .... | .... | ..T. | ..TP | .OTP |
| wind | .... | .... | .... | .... | ..T. | .OT. |
| siren | D.T. | D.T. | ..TP | ..TP | .OTP | .OTP |
| gunshot | .... | .... | ..T. | ..TP | .OTP | .OTP |
| artillery | D.T. | D.T. | ..T. | .OTP | .OTP | .OTP |
| mixed | .... | .... | .... | ..T. | .OT. | .OTP |

### final_A: deployed minus PyTorch

Paired over 1200 clips (deployed ONNX fp32 minus PyTorch); mean [95% bootstrap CI]:

| metric | difference |
|---|---|
| dSI-SNR (dB) | -0.137 [-0.172, -0.107] (max |per-clip difference| 11.044) |
| STOI | -0.0007 [-0.0011, -0.0004] (max |per-clip difference| 0.0841) |
| PESQ | -0.020 [-0.026, -0.015] (max |per-clip difference| 0.776) |

Cell-level agreement (48 cells): ALL-strict verdict differs in **1**, ALL-alt in **2**, any single-target verdict in **4**; largest cell-mean dSI-SNR difference 1.08 dB.

### final_A: INT8 minus fp32 (deployed path)

Paired over 1200 clips (deployed INT8 minus deployed fp32); mean [95% bootstrap CI]:

| metric | difference |
|---|---|
| dSI-SNR (dB) | -0.113 [-0.132, -0.093] (max |per-clip difference| 3.453) |
| STOI | -0.0002 [-0.0006, +0.0001] (max |per-clip difference| 0.0835) |
| PESQ | -0.063 [-0.069, -0.057] (max |per-clip difference| 0.568) |

INT8 ALL-strict PASS 0, ALL-alt PASS 12 (fp32: 0 / 12); ALL verdict differs from fp32 in 2 cells.

## final_B (test-other speakers, all new)

### final_B: DEPLOYED (ONNX fp32) full target matrix

| class | input SNR | n | dSI-SNR | dSNR>15 | out SI-SNR | oSNR>15 | STOI | >0.85 | PESQ | >2.5 | **ALL (strict, dSNR)** | ALL (alt, oSNR) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| engine | -10 | 25 | +12.8 | fail | +2.8 | fail | 0.737 | fail | 1.26 | fail | **fail** | fail |
| engine | -5 | 25 | +12.0 | fail | +7.0 | fail | 0.802 | fail | 1.55 | fail | **fail** | fail |
| engine | +0 | 25 | +11.8 | fail | +11.7 | fail | 0.883 | marg | 1.95 | fail | **fail** | fail |
| engine | +5 | 25 | +10.4 | fail | +15.3 | marg | 0.922 | PASS | 2.44 | marg | **fail** | marg |
| engine | +10 | 25 | +7.7 | fail | +17.5 | PASS | 0.949 | PASS | 2.85 | PASS | **fail** | PASS |
| engine | +15 | 25 | +6.1 | fail | +20.7 | PASS | 0.969 | PASS | 3.20 | PASS | **fail** | PASS |
| helicopter | -10 | 25 | +12.9 | fail | +2.9 | fail | 0.650 | fail | 1.16 | fail | **fail** | fail |
| helicopter | -5 | 25 | +12.4 | fail | +7.3 | fail | 0.778 | fail | 1.44 | fail | **fail** | fail |
| helicopter | +0 | 25 | +10.7 | fail | +10.6 | fail | 0.859 | marg | 1.77 | fail | **fail** | fail |
| helicopter | +5 | 25 | +8.8 | fail | +13.7 | fail | 0.910 | PASS | 2.10 | fail | **fail** | fail |
| helicopter | +10 | 25 | +6.0 | fail | +15.6 | marg | 0.943 | PASS | 2.49 | marg | **fail** | marg |
| helicopter | +15 | 25 | +3.7 | fail | +17.7 | PASS | 0.964 | PASS | 2.99 | PASS | **fail** | PASS |
| drone | -10 | 25 | +12.7 | fail | +2.7 | fail | 0.651 | fail | 1.20 | fail | **fail** | fail |
| drone | -5 | 25 | +10.0 | fail | +4.9 | fail | 0.743 | fail | 1.32 | fail | **fail** | fail |
| drone | +0 | 25 | +9.4 | fail | +9.4 | fail | 0.834 | marg | 1.63 | fail | **fail** | fail |
| drone | +5 | 25 | +7.1 | fail | +12.0 | fail | 0.910 | PASS | 1.95 | fail | **fail** | fail |
| drone | +10 | 25 | +5.6 | fail | +15.3 | marg | 0.927 | PASS | 2.43 | marg | **fail** | marg |
| drone | +15 | 25 | +3.7 | fail | +18.2 | PASS | 0.951 | PASS | 2.91 | PASS | **fail** | PASS |
| wind | -10 | 25 | +9.5 | fail | -0.6 | fail | 0.496 | fail | 1.08 | fail | **fail** | fail |
| wind | -5 | 25 | +8.2 | fail | +3.2 | fail | 0.627 | fail | 1.15 | fail | **fail** | fail |
| wind | +0 | 25 | +7.4 | fail | +7.3 | fail | 0.746 | fail | 1.33 | fail | **fail** | fail |
| wind | +5 | 25 | +7.0 | fail | +11.9 | fail | 0.839 | marg | 1.71 | fail | **fail** | fail |
| wind | +10 | 25 | +5.0 | fail | +14.8 | marg | 0.888 | PASS | 2.05 | fail | **fail** | fail |
| wind | +15 | 25 | +3.0 | fail | +17.3 | PASS | 0.927 | PASS | 2.41 | marg | **fail** | marg |
| siren | -10 | 25 | +21.2 | PASS | +11.3 | fail | 0.908 | PASS | 2.14 | fail | **fail** | fail |
| siren | -5 | 25 | +16.3 | marg | +11.3 | fail | 0.908 | PASS | 2.30 | marg | **marg** | fail |
| siren | +0 | 25 | +14.5 | marg | +14.4 | marg | 0.950 | PASS | 2.90 | PASS | **marg** | marg |
| siren | +5 | 25 | +10.7 | fail | +15.5 | marg | 0.958 | PASS | 2.94 | PASS | **fail** | marg |
| siren | +10 | 25 | +7.6 | fail | +17.3 | PASS | 0.970 | PASS | 3.38 | PASS | **fail** | PASS |
| siren | +15 | 25 | +4.8 | fail | +19.1 | PASS | 0.979 | PASS | 3.64 | PASS | **fail** | PASS |
| gunshot | -10 | 25 | +17.0 | PASS | +7.1 | fail | 0.783 | fail | 1.56 | fail | **fail** | fail |
| gunshot | -5 | 25 | +14.4 | marg | +9.5 | fail | 0.844 | marg | 1.87 | fail | **fail** | fail |
| gunshot | +0 | 25 | +12.4 | fail | +12.4 | fail | 0.901 | PASS | 2.31 | marg | **fail** | fail |
| gunshot | +5 | 25 | +9.8 | fail | +14.7 | marg | 0.938 | PASS | 2.57 | marg | **fail** | marg |
| gunshot | +10 | 25 | +8.1 | fail | +17.7 | PASS | 0.965 | PASS | 3.17 | PASS | **fail** | PASS |
| gunshot | +15 | 25 | +5.6 | fail | +20.0 | PASS | 0.967 | PASS | 3.41 | PASS | **fail** | PASS |
| artillery | -10 | 25 | +18.7 | PASS | +10.4 | fail | 0.873 | PASS | 2.05 | fail | **fail** | fail |
| artillery | -5 | 25 | +17.9 | PASS | +13.8 | fail | 0.918 | PASS | 2.48 | marg | **marg** | fail |
| artillery | +0 | 25 | +15.6 | marg | +15.7 | marg | 0.943 | PASS | 2.72 | PASS | **marg** | marg |
| artillery | +5 | 25 | +13.1 | fail | +18.0 | PASS | 0.961 | PASS | 3.13 | PASS | **fail** | PASS |
| artillery | +10 | 25 | +10.6 | fail | +20.4 | PASS | 0.975 | PASS | 3.41 | PASS | **fail** | PASS |
| artillery | +15 | 25 | +6.1 | fail | +20.3 | PASS | 0.973 | PASS | 3.53 | PASS | **fail** | PASS |
| mixed | -10 | 25 | +11.3 | fail | +1.4 | fail | 0.614 | fail | 1.18 | fail | **fail** | fail |
| mixed | -5 | 25 | +11.4 | fail | +6.5 | fail | 0.707 | fail | 1.36 | fail | **fail** | fail |
| mixed | +0 | 25 | +9.6 | fail | +9.6 | fail | 0.828 | marg | 1.58 | fail | **fail** | fail |
| mixed | +5 | 25 | +7.5 | fail | +12.5 | fail | 0.887 | PASS | 1.86 | fail | **fail** | fail |
| mixed | +10 | 25 | +6.0 | fail | +15.8 | PASS | 0.932 | PASS | 2.34 | fail | **fail** | fail |
| mixed | +15 | 25 | +4.2 | fail | +18.6 | PASS | 0.954 | PASS | 2.85 | PASS | **fail** | PASS |

Cells: 48. ALL strict (dSNR): PASS 0, MARGINAL 4, FAIL 44. ALL alt (oSNR): PASS 12, MARGINAL 8, FAIL 28.

Per input SNR (cells passing ALL, strict / alt): -10 dB: 0/0 of 8, -5 dB: 0/0 of 8, +0 dB: 0/0 of 8, +5 dB: 0/1 of 8, +10 dB: 0/4 of 8, +15 dB: 0/7 of 8

Compact grid, deployed path (letters = target PASSED: D = dSNR>15, O = oSNR>15, T = STOI>0.85, P = PESQ>2.5; '.' = not passed):

| class | -10 | -5 | +0 | +5 | +10 | +15 |
|---|---|---|---|---|---|---|
| engine | .... | .... | .... | ..T. | .OTP | .OTP |
| helicopter | .... | .... | .... | ..T. | ..T. | .OTP |
| drone | .... | .... | .... | ..T. | ..T. | .OTP |
| wind | .... | .... | .... | .... | ..T. | .OT. |
| siren | D.T. | ..T. | ..TP | ..TP | .OTP | .OTP |
| gunshot | D... | .... | ..T. | ..T. | .OTP | .OTP |
| artillery | D.T. | D.T. | ..TP | .OTP | .OTP | .OTP |
| mixed | .... | .... | .... | ..T. | .OT. | .OTP |

### final_B: PyTorch full target matrix (for comparison)

| class | input SNR | n | dSI-SNR | dSNR>15 | out SI-SNR | oSNR>15 | STOI | >0.85 | PESQ | >2.5 | **ALL (strict, dSNR)** | ALL (alt, oSNR) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| engine | -10 | 25 | +12.9 | fail | +2.9 | fail | 0.736 | fail | 1.26 | fail | **fail** | fail |
| engine | -5 | 25 | +12.1 | fail | +7.0 | fail | 0.803 | fail | 1.55 | fail | **fail** | fail |
| engine | +0 | 25 | +11.8 | fail | +11.7 | fail | 0.884 | marg | 1.94 | fail | **fail** | fail |
| engine | +5 | 25 | +10.4 | fail | +15.3 | marg | 0.922 | PASS | 2.45 | marg | **fail** | marg |
| engine | +10 | 25 | +7.8 | fail | +17.5 | PASS | 0.949 | PASS | 2.87 | PASS | **fail** | PASS |
| engine | +15 | 25 | +6.2 | fail | +20.8 | PASS | 0.969 | PASS | 3.21 | PASS | **fail** | PASS |
| helicopter | -10 | 25 | +13.0 | fail | +3.0 | fail | 0.651 | fail | 1.17 | fail | **fail** | fail |
| helicopter | -5 | 25 | +12.4 | fail | +7.3 | fail | 0.777 | fail | 1.45 | fail | **fail** | fail |
| helicopter | +0 | 25 | +10.7 | fail | +10.7 | fail | 0.860 | marg | 1.78 | fail | **fail** | fail |
| helicopter | +5 | 25 | +8.9 | fail | +13.8 | fail | 0.910 | PASS | 2.14 | fail | **fail** | fail |
| helicopter | +10 | 25 | +6.0 | fail | +15.6 | marg | 0.944 | PASS | 2.49 | marg | **fail** | marg |
| helicopter | +15 | 25 | +3.8 | fail | +17.7 | PASS | 0.965 | PASS | 3.04 | PASS | **fail** | PASS |
| drone | -10 | 25 | +13.1 | fail | +3.1 | fail | 0.665 | fail | 1.22 | fail | **fail** | fail |
| drone | -5 | 25 | +10.4 | fail | +5.4 | fail | 0.744 | fail | 1.35 | fail | **fail** | fail |
| drone | +0 | 25 | +9.6 | fail | +9.5 | fail | 0.836 | marg | 1.65 | fail | **fail** | fail |
| drone | +5 | 25 | +7.4 | fail | +12.3 | fail | 0.912 | PASS | 2.02 | fail | **fail** | fail |
| drone | +10 | 25 | +5.9 | fail | +15.7 | marg | 0.930 | PASS | 2.50 | marg | **fail** | marg |
| drone | +15 | 25 | +3.8 | fail | +18.2 | PASS | 0.951 | PASS | 2.98 | PASS | **fail** | PASS |
| wind | -10 | 25 | +9.6 | fail | -0.4 | fail | 0.497 | fail | 1.08 | fail | **fail** | fail |
| wind | -5 | 25 | +8.2 | fail | +3.2 | fail | 0.626 | fail | 1.14 | fail | **fail** | fail |
| wind | +0 | 25 | +7.5 | fail | +7.4 | fail | 0.747 | fail | 1.33 | fail | **fail** | fail |
| wind | +5 | 25 | +7.0 | fail | +11.9 | fail | 0.839 | marg | 1.72 | fail | **fail** | fail |
| wind | +10 | 25 | +5.0 | fail | +14.8 | marg | 0.888 | PASS | 2.05 | fail | **fail** | fail |
| wind | +15 | 25 | +3.0 | fail | +17.3 | PASS | 0.927 | PASS | 2.43 | marg | **fail** | marg |
| siren | -10 | 25 | +21.9 | PASS | +12.0 | fail | 0.910 | PASS | 2.25 | fail | **fail** | fail |
| siren | -5 | 25 | +16.7 | PASS | +11.6 | fail | 0.909 | PASS | 2.33 | marg | **marg** | fail |
| siren | +0 | 25 | +14.6 | marg | +14.6 | marg | 0.950 | PASS | 2.93 | PASS | **marg** | marg |
| siren | +5 | 25 | +10.9 | fail | +15.7 | marg | 0.959 | PASS | 3.00 | PASS | **fail** | marg |
| siren | +10 | 25 | +7.7 | fail | +17.4 | PASS | 0.971 | PASS | 3.43 | PASS | **fail** | PASS |
| siren | +15 | 25 | +4.9 | fail | +19.2 | PASS | 0.979 | PASS | 3.69 | PASS | **fail** | PASS |
| gunshot | -10 | 25 | +17.1 | PASS | +7.2 | fail | 0.779 | fail | 1.54 | fail | **fail** | fail |
| gunshot | -5 | 25 | +14.3 | marg | +9.4 | fail | 0.841 | marg | 1.85 | fail | **fail** | fail |
| gunshot | +0 | 25 | +12.6 | fail | +12.6 | fail | 0.902 | PASS | 2.36 | marg | **fail** | fail |
| gunshot | +5 | 25 | +9.9 | fail | +14.8 | marg | 0.938 | PASS | 2.57 | marg | **fail** | marg |
| gunshot | +10 | 25 | +8.2 | fail | +17.8 | PASS | 0.966 | PASS | 3.18 | PASS | **fail** | PASS |
| gunshot | +15 | 25 | +5.7 | fail | +20.1 | PASS | 0.967 | PASS | 3.43 | PASS | **fail** | PASS |
| artillery | -10 | 25 | +18.9 | PASS | +10.6 | fail | 0.871 | PASS | 2.03 | fail | **fail** | fail |
| artillery | -5 | 25 | +18.2 | PASS | +14.0 | marg | 0.917 | PASS | 2.50 | marg | **marg** | marg |
| artillery | +0 | 25 | +15.9 | marg | +16.0 | marg | 0.943 | PASS | 2.76 | PASS | **marg** | marg |
| artillery | +5 | 25 | +13.1 | fail | +18.0 | PASS | 0.962 | PASS | 3.18 | PASS | **fail** | PASS |
| artillery | +10 | 25 | +10.6 | fail | +20.4 | PASS | 0.975 | PASS | 3.47 | PASS | **fail** | PASS |
| artillery | +15 | 25 | +6.1 | fail | +20.3 | PASS | 0.973 | PASS | 3.57 | PASS | **fail** | PASS |
| mixed | -10 | 25 | +12.1 | fail | +2.2 | fail | 0.615 | fail | 1.20 | fail | **fail** | fail |
| mixed | -5 | 25 | +11.6 | fail | +6.7 | fail | 0.707 | fail | 1.38 | fail | **fail** | fail |
| mixed | +0 | 25 | +9.7 | fail | +9.7 | fail | 0.829 | marg | 1.59 | fail | **fail** | fail |
| mixed | +5 | 25 | +7.6 | fail | +12.5 | fail | 0.887 | PASS | 1.86 | fail | **fail** | fail |
| mixed | +10 | 25 | +6.0 | fail | +15.8 | PASS | 0.932 | PASS | 2.37 | marg | **fail** | marg |
| mixed | +15 | 25 | +4.5 | fail | +19.0 | PASS | 0.955 | PASS | 2.92 | PASS | **fail** | PASS |

Cells: 48. ALL strict (dSNR): PASS 0, MARGINAL 4, FAIL 44. ALL alt (oSNR): PASS 12, MARGINAL 10, FAIL 26.

Per input SNR (cells passing ALL, strict / alt): -10 dB: 0/0 of 8, -5 dB: 0/0 of 8, +0 dB: 0/0 of 8, +5 dB: 0/1 of 8, +10 dB: 0/4 of 8, +15 dB: 0/7 of 8

Compact grid, PyTorch path:

| class | -10 | -5 | +0 | +5 | +10 | +15 |
|---|---|---|---|---|---|---|
| engine | .... | .... | .... | ..T. | .OTP | .OTP |
| helicopter | .... | .... | .... | ..T. | ..T. | .OTP |
| drone | .... | .... | .... | ..T. | ..T. | .OTP |
| wind | .... | .... | .... | .... | ..T. | .OT. |
| siren | D.T. | D.T. | ..TP | ..TP | .OTP | .OTP |
| gunshot | D... | .... | ..T. | ..T. | .OTP | .OTP |
| artillery | D.T. | D.T. | ..TP | .OTP | .OTP | .OTP |
| mixed | .... | .... | .... | ..T. | .OT. | .OTP |

### final_B: deployed minus PyTorch

Paired over 1200 clips (deployed ONNX fp32 minus PyTorch); mean [95% bootstrap CI]:

| metric | difference |
|---|---|
| dSI-SNR (dB) | -0.144 [-0.176, -0.114] (max |per-clip difference| 7.718) |
| STOI | -0.0006 [-0.0011, -0.0003] (max |per-clip difference| 0.0948) |
| PESQ | -0.025 [-0.030, -0.019] (max |per-clip difference| 1.065) |

Cell-level agreement (48 cells): ALL-strict verdict differs in **0**, ALL-alt in **2**, any single-target verdict in **3**; largest cell-mean dSI-SNR difference 0.81 dB.

## Verdict on the statement's targets (deployed path, fresh sets)

| | final_A | final_B | PASS on both |
|---|---|---|---|
| ALL, strict reading (dSNR>15 & STOI>0.85 & PESQ>2.5) | 0 / 48 cells | 0 / 48 | 0 |
| ALL, alt reading (oSNR>15 & STOI>0.85 & PESQ>2.5) | 12 / 48 cells | 12 / 48 | 12 |

Cells that PASS ALL (alt reading) on both fresh sets: engine +10 dB, engine +15 dB, helicopter +15 dB, drone +15 dB, siren +10 dB, siren +15 dB, gunshot +10 dB, gunshot +15 dB, artillery +5 dB, artillery +10 dB, artillery +15 dB, mixed +15 dB

Cells that PASS ALL (strict reading) on both fresh sets: none

**All 48 cells do not pass; the statement's targets are met only in the cells listed above.** Wherever a cell is MARG the data cannot distinguish pass from fail at 25 clips.

## Weapon sets: deployed vs PyTorch by peak level (mean over 40 clips per peak)

| set | peak (dB SPL) | STOI PyTorch | STOI deployed | PESQ PyTorch | PESQ deployed | dSI-SNR PyTorch | dSI-SNR deployed |
|---|---|---|---|---|---|---|---|
| final_A | 110 | 0.939 | 0.939 | 2.70 | 2.70 | +11.7 | +11.6 |
| final_A | 120 | 0.839 | 0.838 | 2.17 | 2.17 | +16.4 | +16.1 |
| final_A | 130 | 0.748 | 0.742 | 1.60 | 1.61 | +18.4 | +17.5 |
| final_A | 140 | 0.678 | 0.674 | 1.51 | 1.49 | +19.9 | +18.1 |
| final_A | 150 | 0.640 | 0.629 | 1.38 | 1.34 | +22.4 | +19.6 |
| final_A | 160 | 0.586 | 0.569 | 1.34 | 1.33 | +25.6 | +21.5 |
| final_B | 110 | 0.926 | 0.926 | 2.74 | 2.72 | +12.1 | +12.1 |
| final_B | 120 | 0.835 | 0.836 | 2.14 | 2.15 | +16.8 | +16.7 |
| final_B | 130 | 0.729 | 0.727 | 1.66 | 1.69 | +19.3 | +18.7 |
| final_B | 140 | 0.663 | 0.658 | 1.45 | 1.61 | +20.8 | +19.1 |
| final_B | 150 | 0.602 | 0.597 | 1.38 | 1.35 | +25.1 | +22.8 |
| final_B | 160 | 0.546 | 0.546 | 1.29 | 1.26 | +25.0 | +22.2 |
