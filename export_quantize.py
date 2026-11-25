"""Export best.pt -> ONNX (static 1x3x192x192) -> INT8 QDQ ONNX (post-training quantization)."""
import argparse, glob, os
import cv2, numpy as np, torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType, quantize_static)
from common import LandmarkNet, S
from dataset import FaceLM


class Reader(CalibrationDataReader):
    def __init__(self, ann, n, calib_dir):
        ds = FaceLM(ann, "val", False)
        step = max(1, len(ds) // n)
        xs = [ds[i][0].numpy()[None] for i in range(0, len(ds), step)][:n]
        for f in sorted(glob.glob(os.path.join(calib_dir, "*"))) if calib_dir else []:
            im = cv2.imread(f)  # pre-cropped DK-camera face crops
            if im is None: continue
            im = cv2.cvtColor(cv2.resize(im, (S, S)), cv2.COLOR_BGR2RGB).astype(np.float32)
            xs.append(((im - 127.5) / 127.5).transpose(2, 0, 1)[None])
        print("calibration samples:", len(xs))
        self.it = iter([{"input": x.astype(np.float32)} for x in xs])

    def get_next(self):
        return next(self.it, None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ann", default="data/annotations.json")
    ap.add_argument("--out_dir", default="export")
    ap.add_argument("--calib_n", type=int, default=300)
    ap.add_argument("--calib_dir", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu")
    model = LandmarkNet(ck["backbone"], pretrained=False)
    model.load_state_dict(ck["ema"]); model.eval()
    fp32 = f"{a.out_dir}/model_fp32.onnx"
    kw = dict(input_names=["input"], output_names=["landmarks"], opset_version=13, do_constant_folding=True)
    dummy = torch.zeros(1, 3, S, S)
    try:
        torch.onnx.export(model, dummy, fp32, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, dummy, fp32, **kw)
    print("saved", fp32)
    pre = f"{a.out_dir}/model_fp32_pre.onnx"
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(fp32, pre); src = pre
    except Exception as e:
        print("pre-process skipped:", e); src = fp32
    int8 = f"{a.out_dir}/model_int8_qdq.onnx"
    quantize_static(src, int8, Reader(a.ann, a.calib_n, a.calib_dir), quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)
    print("saved", int8)