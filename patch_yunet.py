"""Adds an optional YuNet face detector (--yunet model.onnx) to run_demo.py."""
import re
src = open("run_demo.py", newline=None).read()          # universal newlines
if "make_yunet" in src:
    raise SystemExit("already patched")
func = '''def make_yunet(path, score=0.6):
    d = cv2.FaceDetectorYN.create(path, "", (320, 320), score, 0.3, 5000)

    def detect(frame):
        h, w = frame.shape[:2]
        d.setInputSize((w, h))
        _, faces = d.detect(frame)
        return [] if faces is None else [tuple(float(v) for v in r[:4]) for r in faces]
    return detect


'''
m = 'if __name__ == "__main__":'
assert src.count(m) == 1
src = src.replace(m, func + m)
old = "    a = ap.parse_args()"
assert src.count(old) == 1
src = src.replace(old, '    ap.add_argument("--yunet", default=None, help="path to face_detection_yunet_2023mar.onnx (better detector)")\n'
                       '    ap.add_argument("--max_miss", type=int, default=3, help="detector rounds a track may go unmatched before it is dropped")\n' + old)
old = "    trk = Tracker(load_model(a), make_detector(), a)"
assert src.count(old) == 1
src = src.replace(old, "    trk = Tracker(load_model(a), make_yunet(a.yunet) if a.yunet else make_detector(), a)")
old = "self.tracks = [t for t in self.tracks if t.miss < 3]"
assert src.count(old) == 1
src = src.replace(old, "self.tracks = [t for t in self.tracks if t.miss < self.a.max_miss]")
open("run_demo.py", "w", newline="").write(src)
print("patched run_demo.py")
