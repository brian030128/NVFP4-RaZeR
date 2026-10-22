# Part AA: 2,048-window memory probes

The two worst cases of the 256-window runs, at 2,048 windows (PROTOCOL.md part AA). The device peak is nvidia-smi's memory.used every 2 s (part P's sampler); the card has 97,887 MiB (95.6 GiB).

## FlipQuant Qwen3.8-27B 16x64, teacher on the host

`-m calibration.train_map --model qwen3.8-27b --unit 16x64 --epochs 5 --teacher-topk 1000 --fit-windows 2048 --data-root /home/dev/n16k64_campaign/fqrel/tmopt_data --teacher-host --probe-steps 4 --out /home/dev/n16k64_campaign/paper_eval/calib_size/probes/qwen3.8-27b_fq16_2048`

- rc 0, status probe_complete; job 1539 s; device peak 95020 MiB.
- Peak allocated / reserved: 90.52 / 92.06 GiB (the 256-window run with the teacher on the GPU: 91.25 / 92.79 GiB). Host RSS peak: 51.92 GiB.
- Teacher: 2048 windows, 6,283,354,112 bytes on cpu.
- Per phase (GiB allocated): model_load 50.96, data_load 50.96, teacher_precompute 52.65, candidate_packing 55.98, training 90.52.
- Probe: 4 optimizer steps, 13.6 s per step.

## FOCUS Phi-4, micro-batch 8

`-m calibration.train_focus --model phi4-14b --revision 2db69c1c3e91a05d2c64a3185acfbaf36f744e25 --data release --calib-record /home/dev/n16k64_campaign/fqrel/tmopt_data/phi4/calibration/report.json --calib-extension /home/dev/n16k64_campaign/paper_eval/calib_size/windows/phi4-14b_2048.json --epochs 1 --batch 32 --lr-scale 5e-3 --lr-sub 1e-3 --topk 1000 --num-sub 2 --init-q 6 --act fourover6 --act-scope row --seed 42 --ppl-datasets  --seqlen 512 --micro 8 --probe-steps 2 --out /home/dev/n16k64_campaign/paper_eval/calib_size/probes/phi4-14b_focus_2048`

- rc 0, status probe_complete; job 148 s; device peak 96148 MiB.
- Peak allocated / reserved: 91.04 / 93.21 GiB (the 256-window run: 91.04 GiB allocated).
- 2 optimizer steps; step times (s, cumulative): 52, 103; windows 2048.

## Disk

The teacher has no disk cache: train_map keeps it in memory (GPU, or the host with --teacher-host) and FOCUS recomputes the teacher's logits per micro-batch. The windows are cached once per tokenizer and set (flipquant.data's cache):

- `--home--dev--.cache--huggingface--hub--models--Qwen--Qwen3-1.7B--snapshots--70d244cc86ccca08cf5af4e1e306ecf908b1ad5e-24b63593f1d8/calibration_release_472806b28591a3a4.pt`: 9,153,797 bytes
- `--home--dev--.cache--huggingface--hub--models--Qwen--Qwen3-8B--snapshots--b968826d9c46dd6066d109eabc6255188de91218-24b63593f1d8/calibration_release_472806b28591a3a4.pt`: 9,153,797 bytes
- `--home--dev--.cache--huggingface--hub--models--Qwen--Qwen3.8-27B--snapshots--1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0-7ef6def600d7/calibration_release_d84ea8a0ba6a0077.pt`: 9,153,797 bytes
- `--home--dev--.cache--huggingface--hub--models--meta-llama--Llama-3.1-8B-Instruct--snapshots--0e9e39f249a16976918f6564b8830bc894c89659-6ea727c43071/calibration_release_a1c77ea6e2fa1840.pt`: 9,153,797 bytes
- `--home--dev--.cache--huggingface--hub--models--meta-llama--Llama-3.2-1B-Instruct--snapshots--9213176726f574b556790deb65791e0c5aa438b6-6ea727c43071/calibration_release_7434eb003f7220a6.pt`: 9,153,797 bytes
- `--home--dev--.cache--huggingface--hub--models--microsoft--phi-4--snapshots--2db69c1c3e91a05d2c64a3185acfbaf36f744e25-9f5fe260f741/calibration_release_9b118c9b16c2d088.pt`: 9,153,797 bytes
- `--home--dev--.cache--huggingface--hub--models--mistralai--Mistral-7B-Instruct-v0.3--snapshots--c170c708c41dac9275d15a8fff4eca08d52bab71-8be2d4e3de46/calibration_release_f5bfff606ca9e760.pt`: 9,153,797 bytes
- `--home--dev--.cache--huggingface--hub--models--nvidia--NVIDIA-Nemotron-Nano-9B-v2--snapshots--6533e8de2c68e4536bf7c411d7a3ce5734111476-59306981dec2/calibration_release_04959c68a7e48006.pt`: 9,153,797 bytes
- `Qwen--Qwen3-1.7B-24b63593f1d8/calibration_release_24a07f0d1b7c1a60.pt`: 4,576,517 bytes
- `Qwen--Qwen3-1.7B-24b63593f1d8/calibration_release_472806b28591a3a4.pt`: 9,153,797 bytes
- `Qwen--Qwen3-1.7B-24b63593f1d8/calibration_release_b05f021502946062.pt`: 1,145,069 bytes
- `Qwen--Qwen3-8B-24b63593f1d8/calibration_release_24a07f0d1b7c1a60.pt`: 4,576,517 bytes
- `Qwen--Qwen3-8B-24b63593f1d8/calibration_release_472806b28591a3a4.pt`: 9,153,797 bytes
- `Qwen--Qwen3-8B-24b63593f1d8/calibration_release_b05f021502946062.pt`: 1,145,069 bytes
- `Qwen--Qwen3.8-27B-7ef6def600d7/calibration_release_ad1956b044b7f5f5.pt`: 4,576,517 bytes
- `Qwen--Qwen3.8-27B-7ef6def600d7/calibration_release_bd3904fdd5d6480e.pt`: 1,145,069 bytes
- `Qwen--Qwen3.8-27B-7ef6def600d7/calibration_release_d84ea8a0ba6a0077.pt`: 9,153,797 bytes
- `meta-llama--Llama-3.1-8B-Instruct-6ea727c43071/calibration_release_0ac8e91818609669.pt`: 1,145,069 bytes
- `meta-llama--Llama-3.1-8B-Instruct-6ea727c43071/calibration_release_19e8d63a59745d73.pt`: 4,576,517 bytes
- `meta-llama--Llama-3.1-8B-Instruct-6ea727c43071/calibration_release_a1c77ea6e2fa1840.pt`: 9,153,797 bytes
- `meta-llama--Llama-3.2-1B-Instruct-6ea727c43071/calibration_release_04038ca921734be9.pt`: 4,576,517 bytes
- `meta-llama--Llama-3.2-1B-Instruct-6ea727c43071/calibration_release_7434eb003f7220a6.pt`: 9,153,861 bytes
- `meta-llama--Llama-3.2-1B-Instruct-6ea727c43071/calibration_release_f7b75d21a77289be.pt`: 1,145,069 bytes
- `microsoft--phi-4-9f5fe260f741/calibration_release_6c0c464b6f74cccb.pt`: 4,576,517 bytes
- `microsoft--phi-4-9f5fe260f741/calibration_release_9b118c9b16c2d088.pt`: 9,153,797 bytes
- `microsoft--phi-4-9f5fe260f741/calibration_release_d35d7fa67d5b0614.pt`: 1,145,005 bytes
- `mistralai--Mistral-7B-Instruct-v0.3-8be2d4e3de46/calibration_release_2a819f5887677495.pt`: 1,145,069 bytes
- `mistralai--Mistral-7B-Instruct-v0.3-8be2d4e3de46/calibration_release_bfd1649f8e3549da.pt`: 4,576,517 bytes
- `mistralai--Mistral-7B-Instruct-v0.3-8be2d4e3de46/calibration_release_f5bfff606ca9e760.pt`: 9,153,797 bytes
- `nvidia--NVIDIA-Nemotron-Nano-9B-v2-59306981dec2/calibration_release_04959c68a7e48006.pt`: 9,153,797 bytes
- `nvidia--NVIDIA-Nemotron-Nano-9B-v2-59306981dec2/calibration_release_37f2cef261f67cd2.pt`: 1,145,069 bytes
- `nvidia--NVIDIA-Nemotron-Nano-9B-v2-59306981dec2/calibration_release_3d305ffbc5a39dca.pt`: 4,576,517 bytes

Total: 32 files, 192.2 MB (2,048 windows = 9.15 MB, 1,024 = 4.58 MB, 256 = 1.15 MB per tokenizer; the audit's tokenizers, loaded from their snapshot paths, have their own cache keys).
