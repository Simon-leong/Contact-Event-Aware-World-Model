# Contact-Event-Aware World Model (CAEWM)

CAEWM is a visuo-tactile world action model for contact-rich robot manipulation. Given the current scene image, the robot state, the task instruction and, when available, the readings of two optical tactile fingertips, it predicts the next 1.6 s of end-effector actions together with the future fingertip deformation, contact events and video. The model is built on the frozen [Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) video diffusion backbone and extends the HapticWAM code base.

## Installation

The code is tested on Linux with Python 3.11, PyTorch 2.13 and CUDA 12.9. Evaluation runs on one GPU with at least 16 GB of memory, and training uses two 48 GB GPUs.

```bash
git clone --recurse-submodules https://github.com/Simon-leong/Contact-Event-Aware-World-Model.git
cd Contact-Event-Aware-World-Model
conda create -n caewm python=3.11 -y
conda activate caewm
pip install -r requirements/requirements-h100.txt   # or requirements-a100.txt / requirements-5090.txt
pip install -e .
export PYTHONPATH=$PWD
```

The Cosmos-Predict2.5 code is a git submodule. If the repository was cloned without `--recurse-submodules`, run `git submodule update --init`. LPIPS scoring additionally needs `lpips` and `scikit-image`.

## Data Preparation

**Backbone.** Download [`nvidia/Cosmos-Predict2.5-2B`](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B). Only the `robot/action-cond/` folder and `tokenizer.pth` are needed.

**Dataset.** We use the HapticWAM teleoperation dataset `armteam/hapticwam-teleop-dataset` (1,115 episodes over four tasks, about 70 GB) with the official split of 991 training and 124 validation episodes.

1. Unpack the eight task shards into `<data_root>/phantom-episodes/tasks`.
2. Unpack `batch_20260822.tar.zst` into a separate folder and link its 325 episodes into the task folders.
   ```bash
   python tools/intake_recovery.py normalize <batch_folder>
   python tools/intake_recovery.py place <batch_folder> <data_root>/phantom-episodes/tasks
   ```
3. Extract `manifests_v6.tar` into `<data_root>/phantom-episodes`. The resulting `manifests/all.jsonl` should list 991 training and 124 validation episodes.
4. Copy `norm_stats.json` from the dataset and `text_embeddings.pt` from `armteam/hapticwam-teacher` into `<data_root>/phantom-episodes/tasks`.

**Paths.** All locations are read from `configs/paths.yaml`. Create `configs/paths.local.yaml` (ignored by git) with the values for your machine.

```yaml
cosmos_weights_root: /path/to/cosmos-predict2.5-2b
data_root: /path/to/data
runs_root: /path/to/caewm/runs
cosmos_text_embedding_cache: /path/to/data/phantom-episodes/tasks/text_embeddings.pt
```

**Caches.** Training and evaluation read touch and contact events from three caches under `<runs_root>/../cache`, which take about 14 GB in total. The latent cache must be built with the same tactile tokenizer as the world model, so use the released `tactile_ae.pt` when evaluating the released weights.

```bash
python -m phantom.data.tactile_cache
python -m phantom.caewm.build_latent_cache --ae <runs_root>/tactile_ae/tactile_ae.pt
python -m phantom.caewm2.events
```

## Pretrained Models

| Model | Contents | Download |
|:--|:--|:--|
| CAEWM | world model, tactile tokenizer and close cost tables | [Google Drive](https://drive.google.com/file/d/1jRNQtLTaUxg6pDstVIyaBayRuDhCSqMu/view?usp=sharing) |

Unzip `CAEWM_final_weights.zip`, verify the files with `sha256sum -c SHA256SUMS`, and copy its `runs/` folder to your `runs_root`. The world model is then at `<runs_root>/teacher/caewm_final/teacher_020000.pt` and the tactile tokenizer at `<runs_root>/tactile_ae/tactile_ae.pt`.

The HapticWAM baselines are `teacher_v6/teacher_020000.pt` from `armteam/hapticwam-teacher` and `hid_simft/student_001000.pt` from `armteam/hapticwam-student`.

## Training

Train the tactile tokenizer first. This step can be skipped when the released `tactile_ae.pt` is used.

```bash
python -m phantom.train.pretrain_tactile_ae
```

Train the world model from the Cosmos base for 20,000 updates with an effective batch size of 8. On two L40S GPUs this takes about 30 hours. For a single GPU, run `python -m phantom.train.train_teacher` with the same arguments and `--compute configs/compute.caewm_1gpu.yaml`.

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

A newly trained model needs its own decision quantile and close cost tables, which are chosen on every eighth training episode. The released weights already include them.

```bash
CK=<runs_root>/teacher/caewm_final/teacher_020000.pt
TE="python tools/terminal_eval.py --ckpt $CK --data <data_root>/phantom-episodes/tasks --hardware configs/hardware.nuc.yaml --seeds 4 --dump-traj --nfe 2"
mkdir -p calib
for r in median q0.6 q0.7 q0.8; do
  $TE --split train --episode-stride 8 --plan decide --plan-rule $r --touch sensor --out calib/train8_$r.json
done
for m in sensor imagine; do
  for d in -3 -2 -1 0 1 2 3; do
    $TE --split train --episode-stride 8 --plan gt --plan-close-shift $d --touch $m --out calib/${m}_shift$d.json
  done
  $TE --split train --episode-stride 8 --plan gt --plan-close-none --touch $m --out calib/${m}_none.json
  python -m tools.v3_cost_table --runs calib --mode $m --natural calib/train8_*.json --out calib/cost_$m.json
done
```

## Evaluation

All evaluations use the EMA weights on one GPU. `--touch sensor` feeds the measured tactile input, and `--touch imagine` evaluates the same weights without it. The HapticWAM baselines are evaluated with the same tools.

**Action endpoint error** on the 124 validation episodes. This takes about 20 minutes on one L40S. Use `cost_imagine.json` together with `--touch imagine`.

```bash
CK=<runs_root>/teacher/caewm_final/teacher_020000.pt
python tools/terminal_eval.py --ckpt $CK --data <data_root>/phantom-episodes/tasks \
  --hardware configs/hardware.nuc.yaml --split val --seeds 4 --dump-traj --nfe 2 \
  --plan decide --plan-rule q0.8 --k-mean 4 \
  --cost-table <runs_root>/teacher/caewm_final/cost_sensor.json --touch sensor --out terminal_sensor.json
python -m tools.v3_report terminal_sensor.json --out terminal_sensor.report.json
python -m tools.v3_pair terminal_sensor.json <baseline_terminal>.json
```

**Contact prediction and ACC** on the 248 validation windows in `configs/eval/windows_val124x2_seed7.json`.

```bash
python -m tools.cae_imagination_eval --ckpt $CK --touch sensor --out imagination_sensor.json
python -m tools.cae_imagination_eval --ckpt $CK --touch sensor \
  --windows configs/eval/windows_train8x2_seed7.json --out train8_imagination.json
python -m tools.v3_acc_deploy --train-rows train8_imagination_rows.json \
  --val-rows imagination_sensor_rows.json --out acc.json
```

**Event anticipation, force prediction and deformation fields.**

```bash
python -m tools.v2_event_eval --ckpt $CK --out events.json
python -m tools.cae_tactile_eval --ckpt $CK --touch imagine --out tactile_imagine
python -m tools.v3_knn_tactile --ckpt $CK --data <data_root>/phantom-episodes/tasks --out knn.json
```

**Future video.** FVD needs the I3D model `i3d_torchscript.pt` from StyleGAN-V, whose location is read from the environment variable `CAEWM_I3D`.

```bash
python tools/video_dump.py --ckpt $CK --nfe 2 --out video
python tools/video_quality_score.py video
python -m tools.v3_fvd --ckpt $CK --nfe 2 --out fvd.json
```

## Acknowledgement

This code builds on HapticWAM and [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5). We thank their authors for releasing the code, data and models.

## License

This project is released under the [Apache License 2.0](LICENSE). It is derived from the HapticWAM code base, which is distributed under the same license.
