# Multi-Face-Tracking-SDK

A pipeline for **68-point facial landmark detection, multi-face tracking and head-pose estimation**. It covers everything from dataset preparation and training, through ONNX export and INT8 quantization for NPU deployment, to a live PC demo.

The demo detects faces, tracks each one with a persistent ID, smooths the landmarks over time, estimates each face's yaw, and groups faces by the direction they are looking.

---

## How it works

1. **Detector** finds face boxes (Haar cascade by default, or YuNet if you pass `--yunet`). It only runs every few frames.
2. **Landmark model** (`LandmarkNet`: MobileNetV2-0.5 backbone, ReLU6 for NPU friendliness, 136-value regression head) takes a 192×192 aligned face crop and predicts 68 landmarks.
3. **Tracker** matches detections to existing tracks by IoU. After the first frame, each track's crop comes from its own previous landmarks, not the detector box, so tracking stays stable between detector runs.
4. **One Euro filter** smooths the landmarks: heavy smoothing when the face is still, light smoothing when it moves fast. Flip test-time augmentation (TTA) further reduces jitter.
5. **Head pose** (`solvePnP` on 6 landmarks) gives yaw/pitch/roll. Faces are then clustered by yaw to show how many "same direction" groups exist.

---

## Repository contents

| File | Purpose |
|---|---|
| `common.py` | Shared pieces: crop size `S = 192`, region index ranges (jaw/brows/nose/eyes/lips), the left/right landmark `FLIP` index map, the `LandmarkNet` model, `wing_loss`, and `nme_per_point` (error normalised by inter-ocular distance). |
| `prepare_data.py` | Converts **300W**, **WFLW** (98 to 68 points) and **AFLW2000-3D** into one `annotations.json` with 68 points in original-image pixels, plus train/val/aflw splits. |
| `dataset.py` | `FaceLM` PyTorch dataset. Builds a detector-like aligned crop (eye-line rotation, 1.25× margin) with training augmentations: random box shift/scale/rotation, flip with correct left/right swap, brightness, gamma, blur, low-res, noise, occlusion, grayscale. Run it directly with `--viz` to save a grid of sample crops. |
| `train.py` | Training loop: AdamW, warmup plus cosine LR schedule, Wing loss, mixed precision on CUDA, EMA weights, TensorBoard logging. Saves the best checkpoint (by validation NME) to `<out>/best.pt`. |
| `export_quantize.py` | Loads `best.pt`, exports a static 1×3×192×192 FP32 ONNX model, then applies post-training INT8 quantization (QDQ format, per-channel). Calibrates on validation crops and optionally on your own camera crops. |
| `eval_onnx.py` | Compares one or more ONNX models (e.g. FP32 vs INT8) on a split: overall NME, failure rate (NME > 8%) and per-region error. |
| `pose.py` | `head_pose()` returns yaw/pitch/roll in degrees from 68 landmarks via `solvePnP`. `group_directions()` clusters faces by yaw. |
| `run_demo.py` | The live demo: detector, tracker, One Euro filter, flip TTA and head-pose overlay, on a webcam or a single image. |
| `patch_yunet.py` | One-time script that adds the `--yunet` and `--max_miss` options to `run_demo.py`. **The copy of `run_demo.py` in this repo is already patched**, so running it will just print `already patched`. |
| `yunet.onnx` | Pre-downloaded YuNet face detector model, for use with `--yunet`. |
| `viz_train.jpg`, `viz_val.jpg` | Sample output of `dataset.py --viz` for the train and val splits (landmarks drawn on augmented crops). |
| `requirements.txt` | Python dependencies. |

---

## Setup

```bash
git clone https://github.com/harshit24247/Multi-Face-Tracking-SDK
cd Multi-Face-Tracking-SDK
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies: `torch`, `torchvision`, `timm`, `opencv-python`, `numpy`, `scipy`, `onnx`, `onnxruntime`, `tensorboard`, `tqdm`.

Training downloads ImageNet-pretrained MobileNetV2-0.5 weights through `timm` on first run, so it needs internet access. An NVIDIA GPU is strongly recommended for training; the demo runs on CPU.

---

## Running

### Quick start: just run the demo

You need a trained checkpoint (`best.pt`) or an exported ONNX model. The repo does not include one, so either train one (below) or place your own.

```bash
# Webcam, using a checkpoint (default path: runs/wflw_mnv2/best.pt)
python run_demo.py --ckpt runs/exp1/best.pt

# Webcam, using an ONNX model
python run_demo.py --onnx export/model_int8_qdq.onnx

# Single image (writes demo_out.jpg)
python run_demo.py --onnx export/model_fp32.onnx --image photo.jpg

# Better detector (recommended for multiple faces)
python run_demo.py --ckpt runs/exp1/best.pt --yunet yunet.onnx
```

Press **q** in the video window to quit.

**Demo options**

| Flag | Default | Meaning |
|---|---|---|
| `--ckpt` | `runs/wflw_mnv2/best.pt` | PyTorch checkpoint to load |
| `--onnx` | none | Use an ONNX model instead of the checkpoint |
| `--image` | none | Run on one image instead of the webcam |
| `--cam` | `0` | Webcam index |
| `--yunet` | none | Path to the YuNet ONNX model (use `yunet.onnx`) |
| `--det_every` | `5` | Run the face detector every N frames |
| `--max_miss` | `3` | Detector rounds a track can go unmatched before it is dropped |
| `--mincutoff` | `1.0` | One Euro filter: lower means smoother when still |
| `--beta` | `8.0` | One Euro filter: higher means less lag when moving |
| `--yaw_thr` | `15.0` | Yaw gap (degrees) that separates direction groups |
| `--no_tta` | off | Disable flip averaging (faster, noisier) |

The on-screen overlay shows each face's box, ID, yaw, colour-coded landmark regions, FPS, and the count of faces, direction groups and the largest same-direction group.

> **Windows note:** the webcam is opened with `cv2.CAP_DSHOW`, which is Windows-only. On Linux/macOS, change `cv2.VideoCapture(a.cam, cv2.CAP_DSHOW)` in `run_demo.py` to `cv2.VideoCapture(a.cam)`.

### Full pipeline: train your own model

**1. Download the datasets** (not included): 300W, WFLW and AFLW2000-3D, from their official sources.

**2. Build the annotation file**

```bash
python prepare_data.py --w300 path/to/300W --wflw path/to/WFLW --aflw path/to/AFLW2000 --out data/annotations.json
```

Any subset works; pass only the datasets you have. It prints a count per (source, split).

**3. Check the data visually.** This matters especially for the WFLW 98 to 68 point mapping.

```bash
python dataset.py --ann data/annotations.json --split train --viz viz_train.jpg
python dataset.py --ann data/annotations.json --split val   --viz viz_val.jpg
```

**4. Train**

```bash
python train.py --ann data/annotations.json --out runs/exp1 --epochs 120 --bs 64
tensorboard --logdir runs/exp1
```

Other options: `--backbone` (any timm model name, default `mobilenetv2_050`), `--lr`, `--workers`. Each epoch prints loss, overall NME, failure rate and per-region NME. The best checkpoint is saved to `runs/exp1/best.pt`.

**5. Export and quantize to INT8**

```bash
python export_quantize.py --ckpt runs/exp1/best.pt --out_dir export
# optional: calibrate with your own pre-cropped camera faces
python export_quantize.py --ckpt runs/exp1/best.pt --out_dir export --calib_dir path/to/face_crops --calib_n 300
```

Produces `export/model_fp32.onnx`, `export/model_fp32_pre.onnx` and `export/model_int8_qdq.onnx`.

**6. Compare FP32 vs INT8 accuracy**

```bash
python eval_onnx.py --models export/model_fp32.onnx export/model_int8_qdq.onnx --split val
python eval_onnx.py --models export/model_fp32.onnx export/model_int8_qdq.onnx --split aflw --n 2000
```

The `aflw` split is the held-out AFLW2000-3D set, which is useful as a large-pose test set.

**7. Demo with the result**

```bash
python run_demo.py --onnx export/model_int8_qdq.onnx --yunet yunet.onnx
```

---

## Conventions to be aware of

- **Input:** 192×192 RGB, normalised as `(pixel - 127.5) / 127.5`. This must match any firmware or deployment code.
- **Output:** 136 values, which are 68 (x, y) points normalised by the crop size (divide by 1 and multiply by 192 for crop pixels).
- **Landmark layout (0-based):** jaw 0 to 16, brows 17 to 26, nose 27 to 35, eyes 36 to 47, lips 48 to 67.
- **Metric:** NME normalised by inter-ocular distance (outer eye corners, points 36 and 45); a sample counts as a "failure" above 8%.
- The ONNX input name is `input` and the output name is `landmarks`.
