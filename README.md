# CAEWM

CAEWM is a world action model for contact-rich robot manipulation. From the current scene image, the robot state, the task instruction and, when a sensor is fitted, the recent history of two optical tactile fingertips, it jointly predicts the next 1.6 s of end-effector actions, the next 3 s of fingertip deformation and contact events, and the next 3 s of video. It is built on the frozen Cosmos-Predict2.5-2B video diffusion backbone and starts from the open-source HapticWAM code base, which is our baseline.

This branch holds the final version of the model together with the code that trains and evaluates it. The `main` branch keeps the first version. With one set of weights trained once and taken at the last training step, CAEWM reaches a lower action endpoint error than the HapticWAM teacher with tactile input (15.87 mm against 20.05 mm) and than the HapticWAM student without it (16.11 mm against 22.30 mm), and it predicts contact, slip, the ACC signals and future frames better than every released HapticWAM model.

## Method

CAEWM keeps the overall design of HapticWAM. The current image, the future images, the tactile observation, the robot state, the future touch and the actions are laid out as frames of one token sequence and denoised jointly with rectified flow. Only low-rank adapters (LoRA) on the backbone and the newly added modules are trained. The central idea is to treat the contact schedule, the time at which each kind of contact event first happens within the prediction window, as a discrete latent variable. The joint distribution is factorised as

$$p(a,c,v\mid o)=\sum_{\pi}p(\pi\mid o)\,p(a\mid \pi,o)\,p(c,v\mid a,\pi,o),$$

where $o$ is the current observation (image, tactile observation, robot state and instruction), $a$, $c$ and $v$ are the future actions, touch and images, and $\pi$ is the contact schedule. The schedule covers seven events, which are contact onset and slip on each finger, gripper close, object lift and release. The model first settles when contact will happen, then generates the actions that carry out this schedule, and finally predicts the touch and images that these actions produce. The method has three parts.

### Contact schedule as a latent variable

When the gripper closes is often ambiguous, and the distribution of event times is multimodal. Conditioning the actions on the expected event probabilities averages over several possible schedules, and the resulting motion hesitates between them. CAEWM therefore predicts the full distribution of event times, picks one schedule from it with a decision rule, and generates the actions under that schedule.

Event times follow a discrete-time hazard model. The 3 s after the current time are split into 12 bins, and for each event the model predicts

$$h_k=P\big(T\in(k\Delta,(k+1)\Delta]\ \big|\ T>k\Delta\big),\qquad F_k=1-\prod_{j=0}^{k}(1-h_j),$$

where $T$ is the time of the first occurrence of the event, $\Delta=0.25$ s is the bin width, $k=0,\dots,11$ indexes the bins, $h_k$ is the hazard of bin $k$ and $F_k$ is the probability that the event has happened by the end of bin $k$. The hazard model handles events that do not occur within 3 s as right-censored samples and gives a calibrated distribution, which the decision step relies on. The prediction head sits after block 14 of the 28 backbone blocks. Its output feeds the anticipatory contact coupling (ACC) of HapticWAM, which produces the contact gate and the closing veto and turns the prediction into an attention bias for blocks 15 to 28 only. A summary of the predicted probabilities is also added to the future-touch and action tokens. The first half of the network thus forms a prediction of the coming contact, and the second half generates under it.

For each event $e$, the schedule takes the bin with the lowest expected cost

$$\hat{\pi}_e=\arg\min_{j}\sum_{k}p_e(k)\,C_e[j,k],\qquad p_e(k)=F_{e,k}-F_{e,k-1},$$

where $j$ and $k$ range over the 12 bins and the outcome that the event does not occur, $p_e(k)$ is the predicted probability that the event first occurs in bin $k$, and $C_e[j,k]$ is the cost of deciding $j$ when the event actually falls in $k$. The cost is asymmetric. Closing too early leaves the hand above the object, deciding not to close stops the descent, and closing one or two bins late costs little. For the gripper close we measure a cost matrix per task on training episodes. For the other six events we take the $q$ quantile of the event-time distribution, which is the Bayes decision under an asymmetric linear cost. The decided schedule enters the action tokens only, as the signed time from each action frame to each event. Touch and images see the schedule only through the actions generated under it, which matches the conditioning in the factorisation.

Once the schedule is fixed, the action distribution is unimodal, so averaging several samples becomes safe. At deployment one planning call draws $N=4$ samples that share one schedule, decided from their averaged event probabilities, and executes

$$\hat{a}=\frac{1}{N}\sum_{n=1}^{N}a^{(n)},\qquad a^{(n)}\sim p_\theta\big(a\mid\hat{\pi},o\big),$$

where $a^{(n)}$ is the action of the $n$-th sample and $\hat{a}$ the executed action. Without a shared schedule the samples could land on different closing times, and their mean would hesitate again. Forecasts of touch, contact and images are generated under the median schedule, because the median is the best point estimate of when contact happens under an absolute time error.

### Tactile latent space and camera-projected alignment

Touch lives in one causal latent space that serves as both input and output. A small convolutional autoencoder, trained beforehand and then frozen, encodes the last four deformation fields of a fingertip together with their frame differences, and decodes the latent back into the current field. The two fingers are rewritten as

$$z^{+}=\tfrac12\big(z^{L}+z^{R}\big),\qquad z^{-}=\tfrac12\big(z^{L}-z^{R}\big),\qquad \delta_l=\frac{z(t_0+\tau_l)-z(t_0)}{\sigma_l},$$

where $z^{L}$ and $z^{R}$ are the latents of the left and right finger (the right pad is flipped vertically first so that both pads share a physical frame), $z^{+}$ is the common mode, which tells whether the object is held, and $z^{-}$ is the differential mode, which tells whether the load is balanced. The model predicts $\delta_l$, the change of the latent at $\tau_l=0.5l$ s ahead ($l=1,\dots,6$) relative to the current time $t_0$, scaled by the standard deviation $\sigma_l$ of such changes on the training split. A prediction of zero is the persistence forecast, and the decoded prediction is a full 64×80 deformation field.

Touch and image are aligned through the camera. The fingertip position $\mathbf{p}$ in the robot base frame is projected into the image with the calibrated extrinsics $(R,\mathbf{t})$ and intrinsics $\mathbf{K}$, and a Gaussian $G$ with a standard deviation of 1.5 tokens is drawn around the projection on the 16×20 token grid. Four GeoBridge modules between backbone blocks exchange information in both directions,

$$\tilde{\mathbf{u}}^{\text{img}}_{ij}=\mathbf{u}^{\text{img}}_{ij}+G(i,j)\,f_{\text{t}}\big(\mathbf{u}^{\text{tac}}\big),\qquad \tilde{\mathbf{u}}^{\text{tac}}=\mathbf{u}^{\text{tac}}+f_{\text{v}}\Big(\sum_{i,j}\bar{G}(i,j)\,\mathbf{u}^{\text{img}}_{ij}\Big),$$

where $\mathbf{u}^{\text{img}}_{ij}$ is the image token at grid cell $(i,j)$, $\mathbf{u}^{\text{tac}}$ are the tactile tokens, $f_{\text{t}}$ and $f_{\text{v}}$ are small MLPs with zero-initialised output layers, and $\bar{G}$ is $G$ normalised to sum to one. Without tactile input the model relies on the second term to infer touch from what it sees near the fingertips. The rotary position rows of the tactile frames are shifted away from the image rows, so the only spatial link between the two modalities is $G$. For future frames, $G$ is centred on the fingertip projected at that future time, computed from the model's own predicted actions and never from the ground-truth trajectory.

### One model with and without tactile input

HapticWAM distils a separate student for robots without tactile sensors. CAEWM covers both settings with one set of weights. During training the tactile observation frame receives the time step

$$t_{\text{obs}}=\begin{cases}0, & m=1,\\ t, & m=0,\end{cases}\qquad m\sim\mathrm{Bernoulli}(0.5),$$

where $t$ is the time step shared by the generated frames of the sample. With $m=1$ the measured touch is a clean condition. With $m=0$ it is noised and reconstructed together with the future, so the model learns to infer the current touch from the image, the robot state and the instruction. This is the same idea as the per-modality time steps of UniDiffuser.

Actions and the remaining tokens use partly separate parameters. Every LoRA on the backbone is split into a shared part and a part chosen by token type,

$$\mathbf{y}=W\mathbf{x}+B_{\text{s}}A_{\text{s}}\,\mathbf{x}+B_{\kappa(\mathbf{x})}A_{\kappa(\mathbf{x})}\,\mathbf{x},\qquad \kappa(\mathbf{x})\in\{\text{act},\text{non}\},$$

where $W$ is the frozen weight, $A_{\text{s}}$ and $B_{\text{s}}$ form a rank-8 adapter shared by all tokens, and $A_{\kappa(\mathbf{x})}$ and $B_{\kappa(\mathbf{x})}$ form a rank-8 adapter for action tokens or for all other tokens. Each token therefore gets a rank-16 update, half of it shared. Attention still runs over the whole sequence, as in Mixture-of-Transformers, but actions and observations no longer share one set of adapter weights.

Sampling takes two Euler steps and follows the factorisation. The first step starts from noise, reads the event-time distribution, decides the schedule and gives a first estimate of the actions, touch and images, and the ACC gate and veto are read here. The second step keeps the schedule, uses the first action estimate to place the future fingertip projections, and refines everything.

### Training

The loss sums the rectified-flow velocity errors of the action, future image, future touch and tactile observation frames (weights 1.0, 0.3, 1.0 and 0.5, the last only on samples without tactile input), the negative log-likelihood of the event times and a Huber loss on fingertip pad and wrist force changes from the middle prediction head (0.5 each), a focal slip loss and a contact loss on the last-layer future-touch frames (0.5 each), and the event and ACC losses of HapticWAM. The schedule of a training sample is the ground truth for 50% of samples, the model's own median decision for 30% and unknown for the rest. The own decision comes from a gradient-free pass at $t=1$, the same condition as the first sampling step, and the same pass provides the action estimate for the future fingertip projections on half of the samples. Another 15% of samples keep touch and actions at their ground truth and noise only the video.

The model is trained once from the Cosmos base for 20,000 updates with an effective batch of 8 (one sample per GPU, gradient accumulation of 4, two L40S cards), LoRA learning rate 1e-4, 3e-4 for new modules, 500 warm-up steps, cosine decay and an EMA of 0.999, which took about 30 hours. All results use the EMA weights of the last step, with no checkpoint selection on validation. The model has 52.8 M trainable parameters, 31.7 M of them in the adapters. The inference settings, namely the quantile $q$, the per-task close cost matrices and the weight that fuses the ACC contact probability with the predicted contact into the closing veto, are chosen on a training subset (every eighth training episode) and fixed before validation.

## Results

All numbers use the official HapticWAM split, with 991 training and 124 validation episodes, and the released HapticWAM checkpoints. The endpoint probe predicts the 1.6 s of actions that start 1.6 s before the first gripper close of each validation episode and measures the distance between the final end-effector position and the demonstration, over 124 windows with 4 seeds each. The other metrics use the fixed 248 validation windows in [configs/eval/windows_val124x2_seed7.json](configs/eval/windows_val124x2_seed7.json) with 4 seeds each. HapticWAM runs one denoising step as released. CAEWM acts with two steps and the mean of four samples and reports perception after one and two steps. Contact quantities are scored with the HapticWAM scoring functions after converting the decoded fields into the HapticWAM contact package. Confidence intervals come from a paired bootstrap over episodes.

Endpoint error in mm. Lower is better.

| Model | Mean | Median | > 60 mm | Whiteboard | Waffles | Egg | Carton |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| *With tactile input* | | | | | | | |
| HapticWAM teacher (before sim fine-tuning) | 20.05 | 14.2 | 4.2% | 15.70 | **20.15** | 5.21 | 38.67 |
| CAEWM | **15.87** | **11.4** | **2.2%** | **11.85** | 20.56 | **5.20** | **25.75** |
| *Without tactile input* | | | | | | | |
| HapticWAM student | 22.30 | 15.6 | 5.8% | 15.82 | 35.63 | 7.40 | 30.38 |
| CAEWM | **16.11** | **11.4** | **2.4%** | **12.23** | **21.27** | **5.50** | **25.26** |

The paired difference to the teacher is −4.18 mm (95% CI [−6.94, −2.01]) and to the student −6.19 mm ([−8.14, −4.27]). Waffles with tactile input is the one result that does not beat the baseline, at +0.41 mm with an interval of [−1.62, +2.65].

Contact prediction on 248 windows. Higher is better for the IoU and the F1 scores, lower for the errors.

| Model | Steps | Mask IoU | CoP error | Event F1 | Onset F1 | Onset error (ms) |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| *With tactile input* | | | | | | |
| HapticWAM teacher (before sim fine-tuning) | 1 | 0.229 | 0.477 | 0.916 | 0.935 | 334 |
| HapticWAM teacher (after sim fine-tuning) | 1 | 0.181 | 0.531 | 0.778 | 0.663 | 502 |
| CAEWM | 1 | 0.309 | **0.349** | **0.931** | 0.955 | **314** |
| CAEWM | 2 | **0.312** | 0.375 | 0.927 | **0.956** | 325 |
| *Without tactile input* | | | | | | |
| HapticWAM student | 1 | 0.007 | 0.557 | 0.644 | 0.108 | 515 |
| CAEWM | 1 | **0.185** | **0.448** | **0.934** | **0.944** | **310** |
| CAEWM | 2 | 0.184 | 0.486 | 0.929 | 0.937 | 328 |

ACC contact gate and closing veto at the first sampling step, with tactile input and the deployed threshold of 0.5. Higher is better. The second row applies the CAEWM readout, the mean over four samples and the fused veto, to the HapticWAM teacher.

| Model | Readout | Gate AUROC | Gate F1 | Veto AUROC | Veto F1 |
|:--|:--|:--:|:--:|:--:|:--:|
| HapticWAM teacher | one sample, as released | 0.949 | 0.889 | 0.904 | 0.850 |
| HapticWAM teacher | mean of four, fused veto | 0.949 | **0.898** | 0.891 | 0.852 |
| CAEWM | mean of four, fused veto | **0.952** | **0.898** | **0.938** | **0.869** |

Event anticipation within 1 s, with tactile input. Higher is better except for the ECE. HapticWAM predicts neither the close nor the lift.

| Model | Slip AUROC | Slip AP | Slip ECE | Onset AUROC | Close AUROC | Lift AUROC | Release AP |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| HapticWAM teacher (before sim fine-tuning) | 0.763 | 0.204 | 0.066 | 0.843 | – | – | 0.460 |
| CAEWM | **0.943** | **0.451** | **0.024** | **0.897** | 0.959 | 0.948 | **0.697** |

CAEWM also predicts fingertip pad and wrist force changes, which HapticWAM does not. Measured as skill over the persistence forecast, the pad force reaches 0.25, 0.51 and 0.76 at 0.5, 1 and 3 s, and the wrist wrench 0.34, 0.41 and 0.65.

The decoded 64×80 deformation fields reach an IoU of 0.455 and a CRPS of 0.0279 with tactile input. Without tactile input the one-step IoU is 0.295, above two baselines that see the same information as the model, namely holding the model's own estimate of the current touch (0.157) and nearest-neighbour retrieval from the training set on DINOv3 image features and the robot state (0.274). Retrieval still has a slightly lower CRPS (0.0337 against 0.0346 for CAEWM with two steps), and persistence of the measured current field, which the model cannot see in this setting, reaches 0.331.

Future images, averaged over 1, 2 and 3 s, and FVD over 248 windows. Higher is better for PSNR and SSIM, lower for LPIPS and FVD. The gripper region is a 96×96 crop around the true fingertip.

| Model | Steps | PSNR | SSIM | LPIPS | Gripper LPIPS | FVD |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Copy the current frame | – | 19.82 | 0.827 | 0.107 | – | 896.9 |
| HapticWAM teacher (before sim fine-tuning) | 1 | 19.76 | 0.728 | 0.192 | 0.385 | 1145.9 |
| CAEWM | 1 | **20.45** | 0.773 | 0.164 | 0.352 | 966.6 |
| CAEWM | 2 | 20.43 | **0.780** | **0.152** | **0.325** | **852.2** |

The ablation below changes only the schedule or the decision rule on the same weights, with tactile input. Lower is better. The last row feeds the schedule of the demonstration and is an oracle.

| Schedule and execution | Mean (mm) | > 60 mm | Carton | Other three tasks |
|:--|:--:|:--:|:--:|:--:|
| No schedule (all events unknown), one sample | 19.30 | 5.4% | 40.12 | 12.65 |
| Predicted schedule, median, one sample | 18.91 | 5.0% | 39.32 | 12.40 |
| Predicted schedule, q = 0.8, one sample | 16.94 | 3.6% | 32.98 | 11.83 |
| Same, future fingertip projections off | 16.93 | 3.4% | 33.15 | 11.75 |
| Predicted schedule, Bayes close decision, mean of four (deployed) | 15.87 | 2.2% | 25.75 | 12.72 |
| Ground-truth schedule, one sample (oracle) | 15.55 | 3.0% | 28.13 | 11.53 |

Planning takes 428 ms for one two-step sample on an idle L40S, 2128 ms for the deployed mean of four, and about 2.6 s when the forecast under the median schedule is added.

## Known limitations

The waffles endpoint with tactile input stays 0.41 mm above the HapticWAM teacher, and egg is ahead by only 0.01 mm. If the teacher also averages four samples, it reaches 19.72 mm overall, still 3.85 mm behind CAEWM, but 5.05 mm on egg and 19.92 mm on waffles, both ahead of CAEWM. Most of the remaining error comes from the close decision. About 15% of the samples close two or more bins early or decide not to close, and averaging four samples does not change this share, so the error is systematic. The close cost matrices can only be measured on training episodes, where the model reproduces the demonstration even under a shifted schedule, so the measured costs are compressed. On carton, shifting the close by three bins either way changes the cost by only about 1.5 mm. Their shape is reasonable but their scale is not. The Bayes close decision, the mean of four samples and the fused veto were added after comparing the trained weights with the baseline, although their settings come only from training data. The future fingertip projections have no measurable effect on the 16×20 grid with a 1.6 s action chunk, and the split into shared and action-specific adapters has not been ablated in the final configuration. Deployed planning is five to six times slower than a single sample, and the model has more trainable parameters than the HapticWAM teacher (52.8 M against 27.9 M). Carton varies a lot between training runs, by up to about 13 mm in our experience with the same recipe, and the model was trained only once. All results are offline, and the model has not yet been run in closed loop in simulation or on the robot.

## Repository layout

The CAEWM code lives next to the baseline code it builds on. The code calls the contact schedule a plan, so the related flags are `--plan` and `--plan-rule`.

| Path | Contents |
|:--|:--|
| `phantom/model/tactile/tokenizer.py`, `phantom/train/pretrain_tactile_ae.py` | causal tactile autoencoder and its training |
| `phantom/caewm/` | sequence layout, tactile packing, camera projection, GeoBridge, random tactile conditioning, slip and contact heads |
| `phantom/caewm2/events.py`, `windows.py` | event-time and force labels per episode and the training windows that use them |
| `phantom/caewm2/heads.py`, `dit.py`, `rf.py` | event-time and force prediction head after block 14, ACC conditioning of the later blocks, losses |
| `phantom/caewm3/schedule.py` | schedule decisions (quantile, Bayes, shared over samples) and the time-to-event features |
| `phantom/caewm3/bridge.py`, `geometry.py` | GeoBridge with future fingertip projections |
| `phantom/caewm3/group_lora.py` | shared and action-specific LoRA |
| `phantom/caewm3/dit.py`, `rf.py`, `builder.py`, `load.py` | schedule injection, two-step sampling, model assembly and loading |
| `tools/terminal_eval.py`, `tools/v3_report.py`, `tools/v3_pair.py` | endpoint probe, decomposed endpoint metrics and paired bootstrap |
| `tools/v3_cost_table.py`, `tools/v3_train_windows.py`, `tools/v3_acc_deploy.py` | close cost matrices, training-subset windows and the deployed ACC readout |
| `tools/cae_imagination_eval.py`, `tools/cae_tactile_eval.py`, `tools/v2_event_eval.py` | contact package, deformation fields, event anticipation and force skill |
| `tools/v3_knn_tactile.py`, `tools/v3_anchor_shift.py` | retrieval baseline and the size of the future projection shift |
| `tools/video_dump.py`, `tools/video_quality_score.py`, `tools/v3_fvd.py` | future frames, PSNR, SSIM, LPIPS and FVD |
| `tests/test_caewm3.py` | schedule decisions, LoRA routing, projections and sampling |

## Setup

We use a conda environment with Python 3.11 and PyTorch 2.13 for CUDA 12.9. The pinned packages are in `requirements/`, and the repository installs in editable mode on top of them. The Cosmos-Predict2.5 code is a git submodule and is imported as a library. LPIPS scoring needs `lpips` and `scikit-image`, which we keep in a separate environment.

```bash
git clone --recurse-submodules -b caewm-final https://github.com/Simon-leong/Contact-Event-Aware-World-Model.git caewm && cd caewm
pip install -r requirements/requirements-h100.txt   # or the file that matches your machine
pip install -e .
```

The backbone weights come from `nvidia/Cosmos-Predict2.5-2B` (only `robot/action-cond/` and `tokenizer.pth` are needed). The data is the teleoperation dataset `armteam/hapticwam-teleop-dataset`, assembled as described on the `main` branch, which must give 991 training and 124 validation episodes. The released HapticWAM checkpoints used for comparison are in `armteam/hapticwam-teacher` and `armteam/hapticwam-student`. FVD needs the StyleGAN-V I3D model `i3d_torchscript.pt`, whose location is read from `CAEWM_I3D`. All other locations are read from `configs/paths.yaml`, and per-machine values go into the gitignored `configs/paths.local.yaml`.

```yaml
cosmos_weights_root: /path/to/cosmos-predict2.5-2b
data_root: /path/to/data
runs_root: /path/to/caewm/runs
cosmos_text_embedding_cache: /path/to/data/phantom-episodes/tasks/text_embeddings.pt
```

The trained weights are shared with collaborators as a separate archive, laid out like `runs_root`, together with the close cost matrices that belong to them.

## Reproducing the results

The tactile tokenizer, the tactile caches and the event cache are built once.

```bash
export PYTHONPATH=$PWD
python -m phantom.data.tactile_cache
python -m phantom.train.pretrain_tactile_ae
python -m phantom.caewm.build_latent_cache --ae <runs_root>/tactile_ae/tactile_ae.pt
python -m phantom.caewm2.events
```

The world model is trained from the Cosmos base on two GPUs.

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

The inference settings are then chosen on the training subset. The quantile is the one among the median, 0.6, 0.7 and 0.8 with the lowest endpoint error, which was 0.8 for our weights. The close cost matrices come from runs that give the model the demonstration's schedule with only the close moved by $d$ bins, or removed.

```bash
CK=<runs_root>/teacher/caewm_final/teacher_020000.pt
TE="python tools/terminal_eval.py --ckpt $CK --data <data_root>/phantom-episodes/tasks --hardware configs/hardware.nuc.yaml --seeds 4 --dump-traj --nfe 2"
for r in median q0.6 q0.7 q0.8; do $TE --split train --episode-stride 8 --plan decide --plan-rule $r --touch sensor --out costs/train8_$r.json; done
for m in sensor imagine; do
  for d in -3 -2 -1 0 1 2 3; do $TE --split train --episode-stride 8 --plan gt --plan-close-shift $d --touch $m --out costs/${m}_shift$d.json; done
  $TE --split train --episode-stride 8 --plan gt --plan-close-none --touch $m --out costs/${m}_none.json
  python -m tools.v3_cost_table --runs costs --mode $m --natural costs/train8_*.json --out costs/cost_$m.json
done
```

The deployed endpoint probe uses the selected quantile, the Bayes close decision and the mean of four samples. Swap `sensor` for `imagine` to withhold the measured touch.

```bash
$TE --split val --plan decide --plan-rule q0.8 --k-mean 4 --cost-table costs/cost_sensor.json --touch sensor --out terminal_sensor.json
python -m tools.v3_report terminal_sensor.json --out terminal_sensor.report.json
```

Perception is evaluated under the median schedule, which is the default of the evaluation tools. The ACC readout needs the same rows on the training-subset windows to choose the veto fusion weight.

```bash
python -m tools.cae_imagination_eval --ckpt $CK --nfe 1 --touch sensor --out imagination_sensor.json
python -m tools.cae_imagination_eval --ckpt $CK --nfe 1 --touch sensor --windows configs/eval/windows_train8x2_seed7.json --out train8_imagination.json
python -m tools.v3_acc_deploy --train-rows train8_imagination_rows.json --val-rows imagination_sensor_rows.json --out acc.json
python -m tools.cae_tactile_eval --ckpt $CK --touch imagine --seeds 4 --out tactile_imagine
python -m tools.v3_knn_tactile --ckpt $CK --data <data_root>/phantom-episodes/tasks --out knn.json
python -m tools.v2_event_eval --ckpt $CK --out events.json
python tools/video_dump.py --ckpt $CK --nfe 2 --out video_nfe2 && python tools/video_quality_score.py video_nfe2
python -m tools.v3_fvd --ckpt $CK --nfe 2 --touch sensor --out fvd.json
```

The HapticWAM rows come from the same tools pointed at the released checkpoints, and `tools/haptic_imagination_eval.py` gives their official contact package scores.

## License

This code is released under the Apache License 2.0 in [LICENSE](LICENSE). It is derived from the HapticWAM code base, which is distributed under the same license.
