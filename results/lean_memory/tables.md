| configuration | run | checks | rounds / tries | tiles | scoring per pass | dev evaluation per try | native build per evaluation | setup | optimization | peak GPU allocated / reserved | peak host RSS |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DET-FAKE-256x64 | legacy | PASS | 5 / 44 | 7,617 | 69.7 s | 34.7 s | — | 97 s | 31.3 min | 60.9 / 63.4 GiB | 43.7 GiB |
| DET-FAKE-256x64 | lean | PASS | 5 / 44 | 7,617 | 60.5 s | 34.2 s | — | 103 s | 30.1 min | 47.9 / 55.8 GiB | 43.7 GiB |
| DET-NATIVE-256x64 | legacy | PASS | 8 / 63 | 8,385 | 69.5 s | 13.7 s | 0.11 s | 118 s | 23.6 min | 72.1 / 74.6 GiB | 43.9 GiB |
| DET-NATIVE-256x64 | lean | PASS | 8 / 63 | 8,385 | 60.2 s | 13.0 s | 0.15 s | 121 s | 21.6 min | 48.1 / 54.9 GiB | 43.9 GiB |
| DET-FAKE-8x64 | legacy | PASS | 10 / 129 | 3,454 | 70.2 s | 34.9 s | — | 95 s | 86.8 min | 62.0 / 64.6 GiB | 44.1 GiB |
| DET-FAKE-8x64 | lean | PASS | 10 / 129 | 3,454 | 60.8 s | 34.2 s | — | 104 s | 83.7 min | 49.1 / 57.0 GiB | 44.2 GiB |
| DET-NATIVE-8x64 | legacy | PASS | 9 / 114 | 3,801 | 69.5 s | 13.7 s | 0.11 s | 121 s | 36.5 min | 73.3 / 75.8 GiB | 44.2 GiB |
| DET-NATIVE-8x64 | lean | PASS | 9 / 114 | 3,801 | 60.2 s | 13.0 s | 0.15 s | 119 s | 33.8 min | 49.2 / 55.9 GiB | 44.1 GiB |

| configuration | phase | legacy peak GPU allocated | lean peak GPU allocated |
|---|---|---:|---:|
| DET-FAKE-256x64 | model_load | 15.0 GiB | 15.0 GiB |
| DET-FAKE-256x64 | data_load | 15.0 GiB | 15.0 GiB |
| DET-FAKE-256x64 | teacher_precompute | 15.6 GiB | 15.6 GiB |
| DET-FAKE-256x64 | candidate_packing | 25.7 GiB | 18.4 GiB |
| DET-FAKE-256x64 | initial_dev_eval | 42.0 GiB | 29.0 GiB |
| DET-FAKE-256x64 | scoring | 60.9 GiB | 47.9 GiB |
| DET-FAKE-256x64 | dev_evaluation | 42.1 GiB | 29.1 GiB |
| DET-FAKE-256x64 | final_evaluation | 24.9 GiB | 11.9 GiB |
| DET-NATIVE-256x64 | model_load | 15.0 GiB | 15.0 GiB |
| DET-NATIVE-256x64 | data_load | 15.0 GiB | 15.0 GiB |
| DET-NATIVE-256x64 | teacher_precompute | 15.6 GiB | 15.6 GiB |
| DET-NATIVE-256x64 | candidate_packing | 33.1 GiB | 18.4 GiB |
| DET-NATIVE-256x64 | initial_dev_eval | 49.4 GiB | 29.0 GiB |
| DET-NATIVE-256x64 | native_dev_evaluation | 53.2 GiB | 32.9 GiB |
| DET-NATIVE-256x64 | scoring | 72.1 GiB | 48.1 GiB |
| DET-NATIVE-256x64 | dev_evaluation | 53.3 GiB | 32.9 GiB |
| DET-NATIVE-256x64 | final_evaluation | 36.1 GiB | 12.0 GiB |
| DET-FAKE-8x64 | model_load | 15.0 GiB | 15.0 GiB |
| DET-FAKE-8x64 | data_load | 15.0 GiB | 15.0 GiB |
| DET-FAKE-8x64 | teacher_precompute | 15.6 GiB | 15.6 GiB |
| DET-FAKE-8x64 | candidate_packing | 25.8 GiB | 18.4 GiB |
| DET-FAKE-8x64 | initial_dev_eval | 42.0 GiB | 29.0 GiB |
| DET-FAKE-8x64 | scoring | 62.0 GiB | 49.1 GiB |
| DET-FAKE-8x64 | dev_evaluation | 42.8 GiB | 29.9 GiB |
| DET-FAKE-8x64 | final_evaluation | 25.6 GiB | 12.6 GiB |
| DET-NATIVE-8x64 | model_load | 15.0 GiB | 15.0 GiB |
| DET-NATIVE-8x64 | data_load | 15.0 GiB | 15.0 GiB |
| DET-NATIVE-8x64 | teacher_precompute | 15.6 GiB | 15.6 GiB |
| DET-NATIVE-8x64 | candidate_packing | 33.1 GiB | 18.4 GiB |
| DET-NATIVE-8x64 | initial_dev_eval | 49.4 GiB | 29.1 GiB |
| DET-NATIVE-8x64 | native_dev_evaluation | 53.3 GiB | 32.9 GiB |
| DET-NATIVE-8x64 | scoring | 73.3 GiB | 49.2 GiB |
| DET-NATIVE-8x64 | dev_evaluation | 54.1 GiB | 33.7 GiB |
| DET-NATIVE-8x64 | final_evaluation | 36.9 GiB | 12.8 GiB |
