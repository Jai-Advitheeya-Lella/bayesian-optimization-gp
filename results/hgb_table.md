| Method | best 3-fold CV RMSE | test RMSE of chosen config | seeds beating the default CV score | median evals to do so |
|---|---|---|---|---|
| Random search | 0.4852 ± 0.0046 | 0.4736 ± 0.0043 | 1/10 | 15 |
| Optuna TPE | 0.4800 ± 0.0006 | 0.4711 ± 0.0018 | 8/10 | 28 |
| scikit-optimize GP-EI | 0.4786 ± 0.0009 | 0.4702 ± 0.0023 | 10/10 | 17 |
| BO-Thompson (ours) | 0.4786 ± 0.0012 | 0.4710 ± 0.0015 | 9/10 | 29 |
| BO-EI (ours) | 0.4778 ± 0.0012 | 0.4702 ± 0.0014 | 10/10 | 18 |
| scikit-learn defaults (lr=0.1, 31 leaves) | 0.4807 | 0.4708 | - | - |
