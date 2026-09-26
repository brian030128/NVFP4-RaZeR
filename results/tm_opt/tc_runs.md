### Llama-3.1-8B

| unit | TC tiles / TM-OPT tiles | Jaccard | TC native Wiki / C4 | TC − TM-OPT ΔWiki | ΔC4 | acceptable | fake ΔWiki | fake ΔC4 | TC − MR-OPT ΔWiki | ΔC4 | epoch s TM-OPT → TC | selection TM-OPT → TC | peak GPU TM-OPT → TC |
|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| 8x64 | 309,517 / 306,968 | 0.593 | 6.7827 / 9.6795 | +0.00007 ± 0.00143 (n.s.) | +0.00043 ± 0.00103 (n.s.) | **yes** | +0.00068 ± 0.00139 (n.s.) | +0.00084 ± 0.00099 (n.s.) | -0.00451 ± 0.00158 (better) | -0.00873 ± 0.00323 (better) | 35.2 → 22.8 | 13.9 → 9.8 min | 40.8 → 40.8 GiB |
| 16x64 | 201,648 / 200,450 | 0.594 | 6.7865 / 9.6863 | -0.00062 ± 0.00144 (n.s.) | +0.00051 ± 0.00103 (n.s.) | **yes** | +0.00123 ± 0.00130 (n.s.) | +0.00079 ± 0.00109 (n.s.) | -0.00578 ± 0.00171 (better) | -0.00565 ± 0.00194 (better) | 35.2 → 22.9 | 14.0 → 9.8 min | 40.6 → 40.6 GiB |
| 256x64 | 35,365 / 35,346 | 0.652 | 6.8012 / 9.7228 | +0.00081 ± 0.00152 (n.s.) | -0.00027 ± 0.00124 (n.s.) | **yes** | +0.00028 ± 0.00157 (n.s.) | +0.00229 ± 0.00178 (worse) | -0.00524 ± 0.00171 (better) | -0.00376 ± 0.00210 (better) | 35.2 → 22.8 | 14.0 → 9.8 min | 40.5 → 40.5 GiB |

Checks: {"fourover6_native_repeats": true, "tmopt_native_repeats": {"8x64": true, "16x64": true, "256x64": true}, "fourover6_fake_repeats": true}

### Mistral-7B-v0.3

| unit | TC tiles / TM-OPT tiles | Jaccard | TC native Wiki / C4 | TC − TM-OPT ΔWiki | ΔC4 | acceptable | fake ΔWiki | fake ΔC4 | TC − MR-OPT ΔWiki | ΔC4 | epoch s TM-OPT → TC | selection TM-OPT → TC | peak GPU TM-OPT → TC |
|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| 8x64 | 339,444 / 340,045 | 0.557 | 5.4862 / 8.0261 | +0.00003 ± 0.00085 (n.s.) | -0.00015 ± 0.00066 (n.s.) | **yes** | +0.00115 ± 0.00084 (worse) | -0.00103 ± 0.00139 (n.s.) | +0.00045 ± 0.00091 (n.s.) | -0.00091 ± 0.00065 (better) | 32.0 → 19.6 | 12.1 → 7.9 min | 36.2 → 36.2 GiB |
| 16x64 | 224,408 / 223,845 | 0.576 | 5.4951 / 8.0264 | +0.00066 ± 0.00229 (n.s.) | -0.00026 ± 0.00142 (n.s.) | **yes** | +0.00031 ± 0.00094 (n.s.) | +0.00025 ± 0.00074 (n.s.) | -0.00074 ± 0.00236 (n.s.) | -0.00093 ± 0.00121 (n.s.) | 31.9 → 19.7 | 12.0 → 8.0 min | 36.0 → 36.0 GiB |
| 256x64 | 42,522 / 42,615 | 0.659 | 5.4899 / 8.0326 | -0.00104 ± 0.00096 (better) | +0.00010 ± 0.00071 (n.s.) | **yes** | -0.00106 ± 0.00093 (better) | -0.00011 ± 0.00085 (n.s.) | -0.00093 ± 0.00092 (better) | -0.00036 ± 0.00078 (n.s.) | 31.9 → 19.6 | 12.0 → 7.9 min | 36.0 → 36.0 GiB |

Checks: {"fourover6_native_repeats": true, "tmopt_native_repeats": {"8x64": true, "16x64": true, "256x64": true}, "fourover6_fake_repeats": true}

### Phi-4

| unit | TC tiles / TM-OPT tiles | Jaccard | TC native Wiki / C4 | TC − TM-OPT ΔWiki | ΔC4 | acceptable | fake ΔWiki | fake ΔC4 | TC − MR-OPT ΔWiki | ΔC4 | epoch s TM-OPT → TC | selection TM-OPT → TC | peak GPU TM-OPT → TC |
|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| 8x64 | 418,023 / 423,066 | 0.500 | 6.6128 / 10.4954 | +0.00059 ± 0.00120 (n.s.) | -0.00043 ± 0.00071 (n.s.) | **yes** | +0.00034 ± 0.00107 (n.s.) | -0.00031 ± 0.00067 (n.s.) | -0.00369 ± 0.00143 (better) | -0.00249 ± 0.00078 (better) | 63.2 → 39.8 | 23.6 → 15.8 min | 59.7 → 59.7 GiB |
| 16x64 | 287,062 / 281,702 | 0.499 | 6.6161 / 10.5034 | +0.00080 ± 0.00116 (n.s.) | +0.00009 ± 0.00073 (n.s.) | **yes** | -0.00004 ± 0.00106 (n.s.) | -0.00010 ± 0.00073 (n.s.) | -0.00193 ± 0.00145 (better) | -0.00076 ± 0.00078 (n.s.) | 63.1 → 39.8 | 23.6 → 15.8 min | 59.4 → 59.4 GiB |
| 256x64 | 52,811 / 52,772 | 0.597 | 6.6308 / 10.5137 | +0.00039 ± 0.00124 (n.s.) | +0.00006 ± 0.00076 (n.s.) | **yes** | +0.00057 ± 0.00131 (n.s.) | +0.00050 ± 0.00071 (n.s.) | -0.00317 ± 0.00140 (better) | -0.00095 ± 0.00078 (better) | 63.2 → 39.8 | 23.6 → 15.9 min | 59.2 → 59.2 GiB |

Checks: {"fourover6_native_repeats": true, "tmopt_native_repeats": {"8x64": true, "16x64": true, "256x64": true}, "fourover6_fake_repeats": true}

Non-deterministic TM-OPT+TC probe (Llama 8x64, 3 epochs): epoch seconds 21.0 / 20.9 / 20.9 (QAT C1 non-deterministic: 26.9 s).
