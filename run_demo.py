"""PC demo v2: tracked landmarks = landmark-driven ROI + One Euro filter + flip TTA."""
import argparse, time
import cv2, numpy as np
from common import S, REGIONS, FLIP
from dataset import make_crop
from pose import head_pose, group_directions

COLORS = {"jaw": (255, 128, 0), "brows": (255, 0, 200), "nose": (0, 200, 255),
          "eyes": (255, 255, 0), "lips": (0, 0, 255)}  # BGR


def load_model(a):
    if a.onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
        return lambda x: sess.run(None, {"input": x})[0]
    import torch
    from common import LandmarkNet
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu")
    m = LandmarkNet(ck["backbone"], pretrained=False)
    m.load_state_dict(ck["ema"]); m.eval().to(dev)

    def f(x):
        with torch.no_grad():
            return m(torch.from_numpy(x).to(dev)).float().cpu().numpy()
    return f


def alpha(dt, cutoff):
    return 1.0 / (1.0 + (1.0 / (2 * np.pi * cutoff)) / dt)


class OneEuro:
    """Adaptive low-pass: strong smoothing when still, light smoothing when moving fast."""
    def __init__(self, mincutoff, beta, dcutoff=1.0):
        self.mc, self.beta, self.dc, self.t = mincutoff, beta, dcutoff, None

    def __call__(self, x, t, scale):
        if self.t is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return self.x
        dt = max(t - self.t, 1e-3)
        ad = alpha(dt, self.dc)
        self.dx = ad * (x - self.x) / dt + (1 - ad) * self.dx
        a = alpha(dt, self.mc + self.beta * np.abs(self.dx) / scale)  # speed in face-sizes/s
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


def run_net(f, crop_bgr, tta):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None].astype(np.float32)
    p = f(x).reshape(68, 2) * S
    if tta:  # also predict on the mirrored crop and average (reduces noise)
        q = f(np.ascontiguousarray(x[:, :, :, ::-1])).reshape(68, 2) * S
        q[:, 0] = (S - 1) - q[:, 0]
        p = (p + q[FLIP]) / 2
    return p


def box_crop(img, box, margin=1.3):
    """First-frame crop from a detector box (later frames use the landmarks instead)."""
    x, y, w, h = box
    c = np.array([x + w / 2, y + h / 2 + 0.05 * h])
    size = max(w, h) * margin
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), 0, S / size)
    M[:, 2] += [S / 2 - c[0], S / 2 - c[1]]
    return cv2.warpAffine(img, M, (S, S)), M


def pts_box(p):
    mn, mx = p.min(0), p.max(0)
    return (mn[0], mn[1], mx[0] - mn[0], mx[1] - mn[1])


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


class Track:
    def __init__(self, tid, box, a):
        self.id, self.box, self.pts, self.yaw, self.miss = tid, box, None, None, 0
        self.filt = OneEuro(a.mincutoff, a.beta)


class Tracker:
    def __init__(self, f, detect, a):
        self.f, self.detect, self.a = f, detect, a
        self.tracks, self.next_id, self.n = [], 0, 0

    def match(self, boxes):
        used = set()
        for tr in self.tracks:
            tb = pts_box(tr.pts) if tr.pts is not None else tr.box
            best, bi = 0.15, None
            for i, b in enumerate(boxes):
                v = iou(tb, b) if i not in used else 0
                if v > best: best, bi = v, i
            if bi is None:
                tr.miss += 1
            else:
                used.add(bi); tr.miss = 0; tr.box = boxes[bi]
        self.tracks = [t for t in self.tracks if t.miss < self.a.max_miss]
        for i, b in enumerate(boxes):
            if i not in used:
                self.tracks.append(Track(self.next_id, b, self.a)); self.next_id += 1

    def update(self, frame, t):
        self.n += 1
        h, w = frame.shape[:2]
        if not self.tracks or self.n % self.a.det_every == 0:
            self.match(self.detect(frame))
        for tr in list(self.tracks):
            if tr.pts is None:
                crop, M = box_crop(frame, tr.box)
            else:  # same crop rule as training: landmark box x1.25, eye-line rotation
                crop, _, M = make_crop(frame, tr.pts.astype(np.float32), False)
            Minv = cv2.invertAffineTransform(M)
            p = run_net(self.f, crop, not self.a.no_tta)
            pts = p @ Minv[:, :2].T + Minv[:, 2]
            bw, bh = pts_box(pts)[2:]
            if max(bw, bh) < 30 or max(bw, bh) > 1.5 * max(h, w):  # lost the face
                self.tracks.remove(tr); continue
            tr.pts = tr.filt(pts, t, max(bw, bh))
            yaw = head_pose(tr.pts, w, h)[0]
            tr.yaw = yaw if tr.yaw is None else 0.7 * tr.yaw + 0.3 * yaw
        return self.tracks


def draw(frame, tracks, thr):
    for tr in tracks:
        x0, y0, bw, bh = pts_box(tr.pts)
        cv2.rectangle(frame, (int(x0), int(y0)), (int(x0 + bw), int(y0 + bh)), (0, 200, 0), 2)
        cv2.putText(frame, f"ID{tr.id} yaw {tr.yaw:+.0f}", (int(x0), max(15, int(y0) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 2)
        for name, (s, e) in REGIONS.items():
            for px, py in tr.pts[s:e]:
                cv2.circle(frame, (int(px), int(py)), 2, COLORS[name], -1)
    if tracks:
        _, ng, largest = group_directions(np.array([t.yaw for t in tracks]), thr)
        txt = f"faces {len(tracks)} | direction groups {ng} | largest same-direction group {largest}"
    else:
        txt = "no face"
    cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)


def make_detector():
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(frame):
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        return [tuple(float(v) for v in b) for b in casc.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))]
    return detect


def make_yunet(path, score=0.6):
    d = cv2.FaceDetectorYN.create(path, "", (320, 320), score, 0.3, 5000)

    def detect(frame):
        h, w = frame.shape[:2]
        d.setInputSize((w, h))
        _, faces = d.detect(frame)
        return [] if faces is None else [tuple(float(v) for v in r[:4]) for r in faces]
    return detect


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/wflw_mnv2/best.pt")
    ap.add_argument("--onnx", default=None, help="use an ONNX model instead of the checkpoint")
    ap.add_argument("--image", default=None, help="image path; omit to use the webcam")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--yaw_thr", type=float, default=15.0)
    ap.add_argument("--det_every", type=int, default=5, help="run the face detector every N frames")
    ap.add_argument("--mincutoff", type=float, default=1.0, help="lower = smoother when still")
    ap.add_argument("--beta", type=float, default=8.0, help="higher = less lag when moving")
    ap.add_argument("--no_tta", action="store_true", help="disable flip averaging")
    ap.add_argument("--yunet", default=None, help="path to face_detection_yunet_2023mar.onnx (better detector)")
    ap.add_argument("--max_miss", type=int, default=3, help="detector rounds a track may go unmatched before it is dropped")
    a = ap.parse_args()
    trk = Tracker(load_model(a), make_yunet(a.yunet) if a.yunet else make_detector(), a)
    if a.image:
        img = cv2.imread(a.image)
        draw(img, trk.update(img, time.time()), a.yaw_thr)
        cv2.imwrite("demo_out.jpg", img); print("saved demo_out.jpg")
        cv2.imshow("demo", img); cv2.waitKey(0)
    else:
        cap = cv2.VideoCapture(a.cam, cv2.CAP_DSHOW)
        t0 = time.time(); n = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            draw(frame, trk.update(frame, time.time()), a.yaw_thr)
            n += 1
            cv2.putText(frame, f"FPS {n/(time.time()-t0):.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            cv2.imshow("demo (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        cap.release()
    cv2.destroyAllWindows()
