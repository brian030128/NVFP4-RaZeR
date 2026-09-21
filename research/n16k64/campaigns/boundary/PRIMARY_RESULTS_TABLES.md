# Primary results tables

All effects are delta NLL (delta log perplexity); brackets are paired 95% cluster-bootstrap intervals.

## Outcome-blind common support

| Model | Selected tiles | Retained | Coverage | Tiles/global band |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 1,781 | 1,480 | 0.831 | 370 |
| Qwen3-4B | 4,077 | 3,724 | 0.913 | 931 |
| Mistral-7B-v0.3 | 4,179 | 3,912 | 0.936 | 978 |

## Boundary primary endpoints

| Model | Corpus | beta_kappa | Selected−rejected | Weakest selected−nearest rejected | Power status (slope) |
|---|---|---:|---:|---:|---|
| Llama-3.1-8B | C4 | -0.001201 [-0.001658, -0.000797] | -0.001091 [-0.001785, -0.000458] | +0.000694 [-0.000482, +0.002141] | limited_inference |
| Llama-3.1-8B | WikiText | -0.001217 [-0.001611, -0.000809] | -0.000678 [-0.001256, -0.000044] | +0.001320 [+0.000043, +0.002711] | limited_inference |
| Qwen3-4B | C4 | -0.014815 [-0.015505, -0.014140] | -0.014311 [-0.015108, -0.013500] | -0.004591 [-0.005788, -0.003391] | descriptive |
| Qwen3-4B | WikiText | -0.030797 [-0.032610, -0.029070] | -0.029201 [-0.031159, -0.027229] | -0.006012 [-0.008781, -0.003051] | descriptive |
| Mistral-7B-v0.3 | C4 | -0.000286 [-0.000410, -0.000170] | -0.000578 [-0.000924, -0.000262] | -0.000505 [-0.000994, -0.000021] | limited_inference |
| Mistral-7B-v0.3 | WikiText | -0.000451 [-0.000588, -0.000319] | -0.000736 [-0.001086, -0.000390] | +0.000050 [-0.000640, +0.000709] | limited_inference |

## Full-map corruption primary endpoints

| Model | Corpus | beta_p | p=1−p=0 | near p=.50−p=0 | near−random at p=.50 | Slope power |
|---|---|---:|---:|---:|---:|---|
| Llama-3.1-8B | C4 | +0.002830 [+0.001961, +0.003629] | +0.003012 [+0.001871, +0.004180] | +0.001496 [+0.000648, +0.002351] | -0.000752 [-0.001435, -0.000117] | limited_inference |
| Llama-3.1-8B | WikiText | +0.004001 [+0.003110, +0.004869] | +0.003772 [+0.002616, +0.004913] | +0.001607 [+0.000428, +0.002785] | -0.000685 [-0.001465, +0.000135] | limited_inference |
| Qwen3-4B | C4 | +0.054310 [+0.052050, +0.056561] | +0.053230 [+0.050937, +0.055444] | +0.025847 [+0.024442, +0.027266] | -0.010248 [-0.010963, -0.009510] | limited_inference |
| Qwen3-4B | WikiText | +0.112805 [+0.107341, +0.118594] | +0.110018 [+0.104238, +0.116375] | +0.054967 [+0.051004, +0.059419] | -0.016991 [-0.018486, -0.015551] | descriptive |
| Mistral-7B-v0.3 | C4 | +0.002172 [+0.001698, +0.002648] | +0.002009 [+0.001316, +0.002768] | +0.001191 [+0.000279, +0.002417] | -0.000235 [-0.000614, +0.000208] | limited_inference |
| Mistral-7B-v0.3 | WikiText | +0.002927 [+0.002309, +0.003509] | +0.002895 [+0.002126, +0.003612] | +0.000849 [+0.000302, +0.001359] | -0.000478 [-0.000856, -0.000132] | limited_inference |

## Standardized pooled sensitivities

| Experiment | All models | Leave Qwen3-4B out |
|---|---:|---:|
| Boundary beta_kappa | -0.876115 [-0.922347, -0.822872] | -0.822229 [-0.891130, -0.742034] |
| Corruption beta_p | +0.959100 [+0.917997, +0.985754] | +0.939942 [+0.878436, +0.979968] |
