| model | parameters (MEASURED) | ONNX size KB (MEASURED) | MACs per hop, executed (COMPUTED) | of which GRU | MMACs/s at 62.5 hops/s (COMPUTED) | independent counter, total (ratio to ours) | textbook offline MACs/frame (COMPUTED) |
|---|---|---|---|---|---|---|---|
| Tiny | 8,389 | 225.1 | 253,096 | 23,904 | 15.8 | 253,096 (1.0) | 253,476 |
| Small | 24,629 | 287.5 | 532,496 | 95,616 | 33.3 | 532,496 (1.0) | 533,256 |
| Base | 80,725 | 504.7 | 1,501,984 | 382,464 | 93.9 | 1,501,984 (1.0) | 1,503,504 |
