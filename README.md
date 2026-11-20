# Contact-Event-Aware World Model (CAEWM)

Course project for **CS5478 Intelligent Robots: Algorithms and Systems**, National University of Singapore.

CAEWM is a visuo-tactile world action model for contact-rich robot manipulation. Given the current scene image, the robot state, the task instruction and, when available, the readings of two optical tactile fingertips, it predicts the next 1.6 s of end-effector actions together with the future fingertip deformation, contact events and video. The model is built on the frozen [Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) video diffusion backbone, and the code extends the HapticWAM code base, which is the baseline in our report. This repository contains the code to train and evaluate CAEWM and the instructions to reproduce the results in our final report.

## Contents

1. [Repository Structure](#repository-structure)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Data and Pretrained Weights](#data-and-pretrained-weights)
5. [Quick Start](#quick-start)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Acknowledgements](#acknowledgements)
10. [License](#license)

## Repository Structure

```text
.
├── phantom/                 model, data loading and training code
│   ├── caewm/               tactile packing, camera projection, GeoBridge, sampling with and without touch
│   ├── caewm2/              event-time and force prediction head, event labels
│   ├── caewm3/              contact schedule, shared and action-specific LoRA, two-step sampling
│   └── ...                  backbone, data, training loop and scoring code inherited from HapticWAM
├── tools/                   evaluation, inference calibration and weight export scripts
├── configs/                 path, hardware and compute settings and the evaluation window files
├── tests/                   unit tests
├── requirements/            pinned Python packages for different GPU types
├── assets/, third_party/    robot and sensor assets and vendored teleoperation code from HapticWAM
└── cosmos-predict2.5/       Cosmos-Predict2.5 code (git submodule)
```

The final model is built by `phantom/caewm3`, which reuses the modules in `phantom/caewm` and `phantom/caewm2`. The command line flag `--arch caewm3` selects it in training, and every evaluation tool detects it from the checkpoint.

## Requirements

| Item | Requirement |
|:--|:--|
| Operating system | Linux |
| Python and CUDA | Python 3.11, PyTorch 2.13, CUDA 12.9 |
| GPU for evaluation | one NVIDIA GPU with at least 16 GB of memory |
| GPU for training | two 48 GB GPUs (we used NVIDIA L40S), about 30 hours |
| Disk | about 70 GB for the dataset, 14 GB for the caches, 5 GB for the backbone and 0.4 GB for our weights |

## Installation

```bash
git clone --recurse-submodules https://github.com/Simon-leong/Contact-Event-Aware-World-Model.git
cd Contact-Event-Aware-World-Model
conda create -n caewm python=3.11 -y
conda activate caewm
pip install -r requirements/requirements-h100.txt   # or requirements-a100.txt / requirements-5090.txt
pip install -e .
export PYTHONPATH=$PWD
```

If the repository was cloned without `--recurse-submodules`, run `git submodule update --init` before installing. LPIPS scoring of the future video additionally needs `lpips` and `scikit-image`.

## Data and Pretrained Weights

### Backbone

Download [`nvidia/Cosmos-Predict2.5-2B`](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) from Hugging Face. Only the `robot/action-cond/` folder and `tokenizer.pth` are needed.

### Dataset

We use the HapticWAM teleoperation dataset `armteam/hapticwam-teleop-dataset` on Hugging Face (1,115 episodes over four tasks) with the official split of 991 training and 124 validation episodes.

1. Unpack the eight task shards into `<data_root>/phantom-episodes/tasks`.
2. Unpack `batch_20260822.tar.zst` into a separate folder and link its 325 episodes into the task folders.
   ```bash
   python tools/intake_recovery.py normalize <batch_folder>
   python tools/intake_recovery.py place <batch_folder> <data_root>/phantom-episodes/tasks
   ```
3. Extract `manifests_v6.tar` into `<data_root>/phantom-episodes`. The resulting `manifests/all.jsonl` should list 991 training and 124 validation episodes.
4. Copy `norm_stats.json` from the dataset and `text_embeddings.pt` from `armteam/hapticwam-teacher` into `<data_root>/phantom-episodes/tasks`.

### Pretrained weights

| Model | Contents | Download |
|:--|:--|:--|
| CAEWM | world model (`teacher_020000.pt`), tactile tokenizer (`tactile_ae.pt`) and the close cost tables used at deployment | [Google Drive](https://drive.google.com/file/d/1jRNQtLTaUxg6pDstVIyaBayRuDhCSqMu/view?usp=sharing) |

Unzip `CAEWM_final_weights.zip`, check the files with `sha256sum -c SHA256SUMS` inside the unzipped folder, and copy its `runs/` folder to your `runs_root` (see below). The world model is then found at `<runs_root>/teacher/caewm_final/teacher_020000.pt` and the tokenizer at `<runs_root>/tactile_ae/tactile_ae.pt`.

The HapticWAM baselines are `teacher_v6/teacher_020000.pt` and `teacher_v6_simft/teacher_002000.pt` from `armteam/hapticwam-teacher` and `hid_simft/student_001000.pt` from `armteam/hapticwam-student`.

### Path configuration

All locations are read from `configs/paths.yaml`. Create `configs/paths.local.yaml` (ignored by git) with the values for your machine.

```yaml
cosmos_weights_root: /path/to/cosmos-predict2.5-2b
data_root: /path/to/data
runs_root: /path/to/caewm/runs
cosmos_text_embedding_cache: /path/to/data/phantom-episodes/tasks/text_embeddings.pt
```

The episodes are expected under `<data_root>/phantom-episodes/tasks` and the caches under `<runs_root>/../cache`. Every tool also accepts these locations as command line flags.

### Caches

Training and evaluation read touch and contact events from three caches, which are built once. They take about 13 GB, 670 MB and 34 MB. The latent cache must be built with the same tactile tokenizer as the world model, so use the released `tactile_ae.pt` when you evaluate the released weights.

```bash
python -m phantom.data.tactile_cache
python -m phantom.caewm.build_latent_cache --ae <runs_root>/tactile_ae/tactile_ae.pt
python -m phantom.caewm2.events
```

## Quick Start

The unit tests build small versions of the model on a few real episodes, so they need the dataset and the path configuration from the previous section, but not the backbone weights. Four of them also read the event cache and are skipped until it is built. They take about 30 seconds on one GPU.

```bash
python -m pytest tests/test_caewm.py tests/test_caewm2.py tests/test_caewm3.py
```

With the data, the backbone and the released weights in place, the following command reproduces the main action result of the report, the endpoint error on the 124 validation episodes with tactile input. It takes about 20 minutes on one L40S and should report a mean endpoint error of 15.87 mm. With `--touch imagine` and `cost_imagine.json` it evaluates the same weights without tactile input and should report 16.11 mm.

```bash
CK=<runs_root>/teacher/caewm_final/teacher_020000.pt
python tools/terminal_eval.py --ckpt $CK --data <data_root>/phantom-episodes/tasks \
  --hardware configs/hardware.nuc.yaml --split val --seeds 4 --dump-traj --nfe 2 \
  --plan decide --plan-rule q0.8 --k-mean 4 \
  --cost-table <runs_root>/teacher/caewm_final/cost_sensor.json --touch sensor --out terminal_sensor.json
```

Small differences can come from a different GPU or library version.

## Training

Train the tactile tokenizer first. This step can be skipped when the released `tactile_ae.pt` is used.

```bash
python -m phantom.train.pretrain_tactile_ae
```

Then train the world model from the Cosmos base for 20,000 updates with an effective batch size of 8.

```bash
torchrun --standalone --nproc_per_node 2 -m phantom.train.train_teacher --arch caewm3 --sac-mode block_wise \
  --plan --plan-bridge --group-lora \
  --data <data_root>/phantom-episodes/tasks --hardware configs/hardware.nuc.yaml \
  --allow-config-drift --compute configs/compute.caewm.yaml --run-name caewm_final \
  --max-steps 20000 --ckpt-every 1000 --eval-every 1000 \
  --tactile-ae <runs_root>/tactile_ae/tactile_ae.pt --tactile-lat-cache <runs_root>/../cache/tactile_lat \
  --tactile-cache <runs_root>/../cache/tactile --event-cache <runs_root>/../cache/events \
  --touch-forcing-p 0.5 --video-decouple-p 0.15 --loss-video 0.3 \
  --grasp-frac 0.3 --photo-aug 1.0 --wrench-baseline-rows 8 --cond-dropout 0.1 \
  --action-noise-per-strip --no-action-t-max-of-two \
  --lr 1e-4 --lr-new-modules 3e-4 --warmup-steps 500 --ema-decay 0.999 --device cuda
```

On a single GPU, run `python -m phantom.train.train_teacher` with the same arguments and `--compute configs/compute.caewm_1gpu.yaml`, which keeps the effective batch size at 8 through gradient accumulation.

A newly trained model also needs its own decision quantile and close cost tables, both chosen on a training subset (every eighth training episode). The released weights already include them.

```bash
CK=<runs_root>/teacher/caewm_final/teacher_020000.pt
TE="python tools/terminal_eval.py --ckpt $CK --data <data_root>/phantom-episodes/tasks --hardware configs/hardware.nuc.yaml --seeds 4 --dump-traj --nfe 2"
mkdir -p calib

# decision quantile: keep the rule with the lowest endpoint error (q0.8 for the released weights)
for r in median q0.6 q0.7 q0.8; do
  $TE --split train --episode-stride 8 --plan decide --plan-rule $r --touch sensor --out calib/train8_$r.json
done

# close cost tables, with and without tactile input
for m in sensor imagine; do
  for d in -3 -2 -1 0 1 2 3; do
    $TE --split train --episode-stride 8 --plan gt --plan-close-shift $d --touch $m --out calib/${m}_shift$d.json
  done
  $TE --split train --episode-stride 8 --plan gt --plan-close-none --touch $m --out calib/${m}_none.json
  python -m tools.v3_cost_table --runs calib --mode $m --natural calib/train8_*.json --out calib/cost_$m.json
done
```

## Evaluation

All evaluations use the EMA weights and run on one GPU. `--touch sensor` feeds the measured tactile input, and `--touch imagine` evaluates the same weights without it. The HapticWAM baselines are evaluated with the same tools pointed at their released checkpoints.

| Metric | Command | Output |
|:--|:--|:--|
| Action endpoint error | `tools/terminal_eval.py` as in [Quick Start](#quick-start) | `terminal_sensor.json` |
| Decomposed endpoint metrics | `python -m tools.v3_report terminal_sensor.json --out report.json` | time-tolerant endpoint, energy score, minFDE@4 |
| Paired bootstrap against a baseline | `python -m tools.v3_pair terminal_sensor.json <baseline>.json` | 95% confidence intervals |
| Contact package and contact events | `python -m tools.cae_imagination_eval --ckpt $CK --touch sensor --out imagination_sensor.json` | mask IoU, CoP error, event F1, onset F1 |
| Event anticipation and force skill | `python -m tools.v2_event_eval --ckpt $CK --out events.json` | slip, onset, close, lift and release anticipation |
| Deformation fields | `python -m tools.cae_tactile_eval --ckpt $CK --touch imagine --out tactile_imagine` | field IoU, CRPS, skill score |
| Nearest-neighbour baseline | `python -m tools.v3_knn_tactile --ckpt $CK --data <data_root>/phantom-episodes/tasks --out knn.json` | retrieval baseline without tactile input |
| Future video | `python tools/video_dump.py --ckpt $CK --nfe 2 --out video` and `python tools/video_quality_score.py video` | PSNR, SSIM, LPIPS |
| FVD | `python -m tools.v3_fvd --ckpt $CK --nfe 2 --out fvd.json` | FVD with the StyleGAN-V I3D model |

The ACC contact gate and closing veto are read at deployment as the mean over four samples, with the veto fused with the predicted contact. The fusion weight is chosen on the training-subset windows.

```bash
python -m tools.cae_imagination_eval --ckpt $CK --touch sensor \
  --windows configs/eval/windows_train8x2_seed7.json --out train8_imagination.json
python -m tools.v3_acc_deploy --train-rows train8_imagination_rows.json \
  --val-rows imagination_sensor_rows.json --out acc.json
```

FVD needs the I3D model `i3d_torchscript.pt` released with StyleGAN-V. Set the environment variable `CAEWM_I3D` to its location.

## Results

Results on the 124 validation episodes of the HapticWAM dataset. The HapticWAM column uses the released teacher with tactile input and the released student without it.

| Metric | Tactile input | HapticWAM | CAEWM |
|:--|:--:|:--:|:--:|
| Endpoint error (mm), lower is better | yes | 20.05 | **15.87** |
| Endpoint error (mm), lower is better | no | 22.30 | **16.11** |
| Contact mask IoU | yes | 0.229 | **0.309** |
| Contact onset F1 | no | 0.108 | **0.944** |
| Slip anticipation AUROC within 1 s | yes | 0.763 | **0.943** |
| ACC closing veto AUROC | yes | 0.904 | **0.938** |
| Future video FVD, lower is better | yes | 1145.9 | **852.2** |

## Acknowledgements

This code builds on the HapticWAM code base and on [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5). We thank their authors for releasing the code, data and models.

## License

This code is released under the Apache License 2.0, see [LICENSE](LICENSE). It is derived from the HapticWAM code base, which is distributed under the same license.
