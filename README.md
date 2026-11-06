# Contact-Event-Aware World Model (CAEWM)

This repository contains the code for CAEWM, a visuo-tactile world action model for contact-rich robot manipulation. Given the current scene image, the robot state, the task instruction and, when available, the readings of two optical tactile fingertips, CAEWM predicts the next 1.6 s of end-effector actions together with the future fingertip deformation, contact events and video. The model is built on the frozen [Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) video diffusion backbone, and the code base extends [HapticWAM](https://huggingface.co/armteam), which serves as our baseline.

The `caewm-final` branch holds the final version of the model. The `main` branch holds an earlier version and is kept for reference.

## Installation

The code is tested with Python 3.11, PyTorch 2.13 and CUDA 12.9. The Cosmos-Predict2.5 code is included as a git submodule and imported as a library.

```bash
git clone --recurse-submodules -b caewm-final https://github.com/Simon-leong/Contact-Event-Aware-World-Model.git
cd Contact-Event-Aware-World-Model
conda create -n caewm python=3.11 -y && conda activate caewm
pip install -r requirements/requirements-h100.txt   # or the file that matches your GPU
pip install -e .
export PYTHONPATH=$PWD
```

LPIPS scoring additionally needs `lpips` and `scikit-image`, which we install in a separate environment.

## Data Preparation

**Backbone.** Download `nvidia/Cosmos-Predict2.5-2B` from Hugging Face. Only the `robot/action-cond/` folder and `tokenizer.pth` are needed.

**Dataset.** We use the teleoperation dataset `armteam/hapticwam-teleop-dataset` (1,115 episodes over four tasks) with the official HapticWAM split of 991 training and 124 validation episodes.

1. Unpack the eight task shards into `<data_root>/phantom-episodes/tasks`.
2. Unpack `batch_20260822.tar.zst` into a separate folder and link its 325 episodes into the task folders.
   ```bash
   python tools/intake_recovery.py normalize <batch_folder>
   python tools/intake_recovery.py place <batch_folder> <data_root>/phantom-episodes/tasks
   ```
3. Extract `manifests_v6.tar` into `<data_root>/phantom-episodes`. The resulting `manifests/all.jsonl` should list 991 training and 124 validation episodes.
4. Copy `norm_stats.json` from the dataset and `text_embeddings.pt` from `armteam/hapticwam-teacher` into `<data_root>/phantom-episodes/tasks`.

**Paths.** All locations are read from `configs/paths.yaml`. Put the values for your machine into `configs/paths.local.yaml`, which is ignored by git.

```yaml
cosmos_weights_root: /path/to/cosmos-predict2.5-2b
data_root: /path/to/data
runs_root: /path/to/caewm/runs
cosmos_text_embedding_cache: /path/to/data/phantom-episodes/tasks/text_embeddings.pt
```

**Caches.** Training and evaluation read touch and contact events from three caches, which are built once. The tactile field cache takes about 13 GB, the latent cache about 670 MB and the event cache about 34 MB. The latent cache must be built with the same tactile tokenizer as the world model, so use the released `tactile_ae.pt` when you evaluate the released weights.

```bash
python -m phantom.data.tactile_cache
python -m phantom.caewm.build_latent_cache --ae <runs_root>/tactile_ae/tactile_ae.pt
python -m phantom.caewm2.events
```

## Pretrained Weights

| Model | Contents | Download |
|:--|:--|:--|
| CAEWM | world model (`teacher_020000.pt`), tactile tokenizer (`tactile_ae.pt`) and the close cost tables used at deployment | [Google Drive](https://drive.google.com/file/d/1jRNQtLTaUxg6pDstVIyaBayRuDhCSqMu/view?usp=sharing) |

Unzip `CAEWM_final_weights.zip` and copy its `runs/` folder to your `runs_root`. The world model is then found at `<runs_root>/teacher/caewm_final/teacher_020000.pt` and the tokenizer at `<runs_root>/tactile_ae/tactile_ae.pt`. The archive contains its own README and SHA256 checksums.

The HapticWAM checkpoints used for comparison are `teacher_v6/teacher_020000.pt` and `teacher_v6_simft/teacher_002000.pt` from `armteam/hapticwam-teacher` and `hid_simft/student_001000.pt` from `armteam/hapticwam-student`.

## Training

Train the tactile tokenizer first. This step can be skipped when the released `tactile_ae.pt` is used.

```bash
python -m phantom.train.pretrain_tactile_ae
```

Then train the world model from the Cosmos base. We trained for 20,000 updates with an effective batch size of 8 on two 48 GB L40S GPUs, which took about 30 hours.

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

For a single GPU, use `configs/compute.caewm_1gpu.yaml`, which keeps the effective batch size at 8 through gradient accumulation.

## Inference Calibration

The deployed model needs a decision quantile and per-task close cost tables, both chosen on a training subset (every eighth training episode). The released weights already include the cost tables, so this step is only needed after training a new model.

```bash
CK=<runs_root>/teacher/caewm_final/teacher_020000.pt
TE="python tools/terminal_eval.py --ckpt $CK --data <data_root>/phantom-episodes/tasks --hardware configs/hardware.nuc.yaml --seeds 4 --dump-traj --nfe 2"

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

All evaluations use the EMA weights and run on a single GPU. Use `--touch sensor` to feed the measured tactile input and `--touch imagine` to evaluate the same weights without it.

**Action endpoint error** on the 124 validation episodes in the deployed setting.

```bash
$TE --split val --plan decide --plan-rule q0.8 --k-mean 4 --cost-table <runs_root>/teacher/caewm_final/cost_sensor.json \
  --touch sensor --out terminal_sensor.json
python -m tools.v3_report terminal_sensor.json --out terminal_sensor.report.json
python -m tools.v3_pair terminal_sensor.json <baseline_terminal.json>   # paired bootstrap against a baseline
```

**Contact prediction, ACC and event anticipation** on the 248 validation windows in `configs/eval/windows_val124x2_seed7.json`.

```bash
python -m tools.cae_imagination_eval --ckpt $CK --nfe 1 --touch sensor --out imagination_sensor.json
python -m tools.cae_imagination_eval --ckpt $CK --nfe 1 --touch sensor \
  --windows configs/eval/windows_train8x2_seed7.json --out train8_imagination.json
python -m tools.v3_acc_deploy --train-rows train8_imagination_rows.json \
  --val-rows imagination_sensor_rows.json --out acc.json
python -m tools.v2_event_eval --ckpt $CK --out events.json
```

**Deformation fields** and the nearest-neighbour baseline.

```bash
python -m tools.cae_tactile_eval --ckpt $CK --touch imagine --seeds 4 --out tactile_imagine
python -m tools.v3_knn_tactile --ckpt $CK --data <data_root>/phantom-episodes/tasks --out knn.json
```

**Future video.** FVD uses the StyleGAN-V I3D model `i3d_torchscript.pt`, whose location is read from the environment variable `CAEWM_I3D`.

```bash
python tools/video_dump.py --ckpt $CK --nfe 2 --out video_nfe2
python tools/video_quality_score.py video_nfe2
python -m tools.v3_fvd --ckpt $CK --nfe 2 --touch sensor --out fvd.json
```

The HapticWAM baselines are evaluated with the same tools pointed at the released HapticWAM checkpoints, and `tools/haptic_imagination_eval.py` gives their official contact package scores.

## Results

Results on the validation split of the HapticWAM dataset. HapticWAM uses the released teacher with tactile input and the released student without it.

| Metric | Tactile input | HapticWAM | CAEWM |
|:--|:--:|:--:|:--:|
| Endpoint error (mm), lower is better | ✓ | 20.05 | **15.87** |
| Endpoint error (mm), lower is better | ✗ | 22.30 | **16.11** |
| Contact mask IoU | ✓ | 0.229 | **0.309** |
| Contact onset F1 | ✗ | 0.108 | **0.944** |
| Slip anticipation AUROC within 1 s | ✓ | 0.763 | **0.943** |
| ACC closing veto AUROC | ✓ | 0.904 | **0.938** |
| Future video FVD, lower is better | ✓ | 1145.9 | **852.2** |

## Acknowledgements

This code builds on [HapticWAM](https://huggingface.co/armteam) and [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5). We thank the authors for releasing their code, data and models.

## Citation

If you find this code useful, please cite our paper. The BibTeX entry will be added once the paper is public.

## License

This code is released under the Apache License 2.0 in [LICENSE](LICENSE). It is derived from the HapticWAM code base, which is distributed under the same license.
