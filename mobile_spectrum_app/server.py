from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import argparse
import base64
import cgi
import html
import io
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent

CALIBRATION_FILE = DATA_DIR / "calibration_coeffs.npy"
REFERENCE_FILE = DATA_DIR / "reference_spectrum_no_leaf.npz"

# Same crop and extraction settings from the final_model_imgs notebooks.
X_START = 1600
X_END = 2700
Y_START = 1500
Y_END = 2000

X2_START = 380
X2_END = 900
Y2_START = 260
Y2_END = 390

Y_START_LINE = 45
Y_END_LINE = 110

X_START_LINE = 20
X_END_LINE = 500

EPSILON = 1e-9

LOADED_COEFFS = np.load(CALIBRATION_FILE)
REFERENCE_DATA = np.load(REFERENCE_FILE)


def decode_image(upload_bytes):
    image_array = np.frombuffer(upload_bytes, np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not read the uploaded image. Use a JPG or PNG spectrum photo.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def check_bounds(width, height, left, right, top, bottom, label):
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(
            f"{label} is outside the image. Needed x={left}:{right}, y={top}:{bottom}, "
            f"but image area is {width}x{height}."
        )


def extract_spectrum_from_rgb(image_rgb):
    height, width = image_rgb.shape[:2]
    check_bounds(width, height, X_START, X_END, Y_START, Y_END, "First crop")

    roi = image_rgb[Y_START:Y_END, X_START:X_END]

    roi_height, roi_width = roi.shape[:2]
    check_bounds(roi_width, roi_height, X2_START, X2_END, Y2_START, Y2_END, "Second crop")

    roi2 = roi[Y2_START:Y2_END, X2_START:X2_END]
    roi2_gray = cv2.cvtColor(roi2, cv2.COLOR_RGB2GRAY)

    gray_height, gray_width = roi2_gray.shape[:2]
    check_bounds(
        gray_width,
        gray_height,
        X_START_LINE,
        X_END_LINE,
        Y_START_LINE,
        Y_END_LINE,
        "Extraction ROI",
    )

    selected_region = roi2_gray[Y_START_LINE:Y_END_LINE, X_START_LINE:X_END_LINE]
    intensity = np.mean(selected_region, axis=0)
    x_pixels = np.arange(X_START_LINE, X_END_LINE)
    wavelength = np.polyval(LOADED_COEFFS, x_pixels)

    return {
        "wavelength": wavelength,
        "intensity": intensity,
        "roi": roi,
        "roi2": roi2,
        "roi2_gray": roi2_gray,
    }


def sorted_reference(reference_upload_bytes=None):
    if reference_upload_bytes:
        reference_rgb = decode_image(reference_upload_bytes)
        reference = extract_spectrum_from_rgb(reference_rgb)
        wavelength_ref = reference["wavelength"]
        intensity_ref = reference["intensity"]
    else:
        wavelength_ref = REFERENCE_DATA["wavelength_ref"]
        intensity_ref = REFERENCE_DATA["intensity_ref"]

    idx = np.argsort(wavelength_ref)
    return wavelength_ref[idx], intensity_ref[idx], idx


def mean_band(wavelength, values, low, high):
    mask = (wavelength >= low) & (wavelength <= high)
    if not np.any(mask):
        return 0.0
    return float(np.mean(values[mask]))


def classify_plant(wavelength, absorbance):
    blue_abs = mean_band(wavelength, absorbance, 430, 495)
    green_abs = mean_band(wavelength, absorbance, 495, 570)
    red_abs = mean_band(wavelength, absorbance, 570, 650)
    mean_abs = float(np.mean(absorbance))
    max_abs = float(np.max(absorbance))
    chlorophyll_index = (blue_abs + red_abs) / (green_abs + EPSILON)

    # Simple rule-based estimate from this spectrometer response:
    # healthy green leaves should absorb more strongly in blue/red than green.
    is_healthy = chlorophyll_index >= 2.0 and mean_abs >= 0.25

    if is_healthy:
        verdict = "Healthy"
        explanation = "Blue and red absorbance are high relative to green, matching a chlorophyll-rich leaf response."
    else:
        verdict = "Not healthy"
        explanation = "The absorbance pattern does not show a strong blue/red chlorophyll response compared with green."

    return {
        "verdict": verdict,
        "isHealthy": is_healthy,
        "explanation": explanation,
        "meanAbsorbance": mean_abs,
        "maxAbsorbance": max_abs,
        "blueAbsorbance": blue_abs,
        "greenAbsorbance": green_abs,
        "redAbsorbance": red_abs,
        "chlorophyllIndex": float(chlorophyll_index),
    }


def fig_to_base64(fig):
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def plot_absorbance(wavelength, absorbance, verdict):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(wavelength, absorbance, color="#0f766e", linewidth=2.5)
    ax.axvspan(430, 495, color="#2563eb", alpha=0.08)
    ax.axvspan(495, 570, color="#16a34a", alpha=0.08)
    ax.axvspan(570, 650, color="#dc2626", alpha=0.08)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Relative absorbance")
    ax.set_title(f"Relative Absorbance Spectrum - {verdict}")
    ax.grid(True, alpha=0.25)
    return fig_to_base64(fig)


def plot_roi_box(roi2_gray):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.imshow(roi2_gray, cmap="gray")
    rect = Rectangle(
        (X_START_LINE, Y_START_LINE),
        X_END_LINE - X_START_LINE,
        Y_END_LINE - Y_START_LINE,
        linewidth=2,
        edgecolor="#ef4444",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Fixed Extraction ROI")
    return fig_to_base64(fig)


def analyze_upload(leaf_bytes, reference_bytes=None):
    leaf_rgb = decode_image(leaf_bytes)
    leaf = extract_spectrum_from_rgb(leaf_rgb)

    wavelength_sorted, intensity_ref, idx = sorted_reference(reference_bytes)
    intensity_leaf = leaf["intensity"][idx]

    transmission = intensity_leaf / (intensity_ref + EPSILON)
    absorbance = -np.log10(transmission + EPSILON)
    health = classify_plant(wavelength_sorted, absorbance)

    return {
        "health": health,
        "absorbancePlot": plot_absorbance(wavelength_sorted, absorbance, health["verdict"]),
        "roiPlot": plot_roi_box(leaf["roi2_gray"]),
        "wavelength": wavelength_sorted.round(3).tolist(),
        "absorbance": absorbance.round(5).tolist(),
        "transmission": transmission.round(5).tolist(),
    }


def page_html():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>Plant Spectrum Health</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f3f6f3;
      --panel: #ffffff;
      --text: #17211b;
      --muted: #5d6b63;
      --line: #dbe5dd;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --warn: #b91c1c;
      --ok: #15803d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      min-height: 100vh;
      overflow-x: hidden;
    }
    .creeper {
      position: fixed;
      top: 0;
      bottom: 0;
      width: min(24vw, 190px);
      height: 100vh;
      color: var(--accent);
      pointer-events: none;
      z-index: 0;
      opacity: 0.78;
    }
    .creeper.left {
      left: 0;
      transform: translateX(18px);
    }
    .creeper.right {
      right: 0;
      transform: translateX(-18px) scaleX(-1);
    }
    main {
      position: relative;
      z-index: 1;
      width: min(100%, 780px);
      margin: 0 auto;
      padding: 18px 14px 28px;
    }
    header {
      padding: 18px 2px 14px;
    }
    h1 {
      margin: 0;
      font-size: clamp(2rem, 9vw, 4rem);
      line-height: 0.95;
      letter-spacing: 0;
    }
    .subtitle {
      margin: 12px 0 0;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.45;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin: 12px 0;
      box-shadow: 0 8px 28px rgba(20, 45, 29, 0.08);
    }
    label {
      display: block;
      font-weight: 700;
      margin-bottom: 8px;
    }
    .filebox {
      display: block;
      width: 100%;
      padding: 14px;
      border: 1px dashed #9ab5a5;
      border-radius: 8px;
      background: #fbfdfb;
      color: var(--muted);
    }
    input[type="file"] {
      width: 100%;
      font-size: 0.95rem;
    }
    button {
      width: 100%;
      min-height: 48px;
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: white;
      font-size: 1rem;
      font-weight: 800;
      margin-top: 12px;
    }
    button:active { background: var(--accent-dark); }
    .hint {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 0.88rem;
      line-height: 1.35;
    }
    .status {
      display: none;
      padding: 12px;
      border-radius: 8px;
      margin-top: 12px;
      background: #eef7f4;
      color: var(--accent-dark);
      font-weight: 700;
    }
    .error {
      display: none;
      padding: 12px;
      border-radius: 8px;
      margin-top: 12px;
      background: #fef2f2;
      color: var(--warn);
      font-weight: 700;
    }
    .verdict {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }
    .badge {
      display: inline-flex;
      width: fit-content;
      align-items: center;
      min-height: 34px;
      padding: 6px 10px;
      border-radius: 8px;
      font-weight: 900;
      color: white;
      background: var(--ok);
    }
    .badge.bad { background: var(--warn); }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      min-height: 76px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 700;
    }
    .metric strong {
      display: block;
      margin-top: 6px;
      font-size: 1.15rem;
    }
    img {
      display: block;
      width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: white;
    }
    .results { display: none; }
    @media (max-width: 760px) {
      .creeper {
        width: 118px;
        opacity: 0.24;
      }
      .creeper.left { transform: translateX(-14px); }
      .creeper.right { transform: translateX(14px) scaleX(-1); }
    }
    @media (min-width: 640px) {
      main { padding: 26px 20px 40px; }
      .panel { padding: 18px; }
      .metric-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <svg class="creeper left" viewBox="0 0 180 900" preserveAspectRatio="none" aria-hidden="true">
    <path d="M82 -20 C38 70 136 132 74 222 C22 296 118 350 66 438 C16 522 128 590 64 682 C20 746 94 812 70 930" fill="none" stroke="currentColor" stroke-width="9" stroke-linecap="round"/>
    <g fill="currentColor">
      <ellipse cx="58" cy="82" rx="18" ry="36" transform="rotate(-38 58 82)"/>
      <ellipse cx="104" cy="138" rx="16" ry="34" transform="rotate(42 104 138)"/>
      <ellipse cx="56" cy="242" rx="19" ry="38" transform="rotate(-46 56 242)"/>
      <ellipse cx="112" cy="326" rx="17" ry="34" transform="rotate(42 112 326)"/>
      <ellipse cx="48" cy="462" rx="18" ry="37" transform="rotate(-40 48 462)"/>
      <ellipse cx="106" cy="566" rx="17" ry="35" transform="rotate(38 106 566)"/>
      <ellipse cx="44" cy="700" rx="19" ry="38" transform="rotate(-42 44 700)"/>
      <ellipse cx="94" cy="804" rx="16" ry="34" transform="rotate(40 94 804)"/>
    </g>
    <path d="M76 118 C118 96 132 72 150 38 M74 278 C110 250 126 220 152 190 M72 518 C116 488 136 448 158 410 M70 742 C104 714 130 678 154 636" fill="none" stroke="currentColor" stroke-width="5" stroke-linecap="round" opacity="0.85"/>
  </svg>
  <svg class="creeper right" viewBox="0 0 180 900" preserveAspectRatio="none" aria-hidden="true">
    <path d="M82 -20 C38 70 136 132 74 222 C22 296 118 350 66 438 C16 522 128 590 64 682 C20 746 94 812 70 930" fill="none" stroke="currentColor" stroke-width="9" stroke-linecap="round"/>
    <g fill="currentColor">
      <ellipse cx="58" cy="82" rx="18" ry="36" transform="rotate(-38 58 82)"/>
      <ellipse cx="104" cy="138" rx="16" ry="34" transform="rotate(42 104 138)"/>
      <ellipse cx="56" cy="242" rx="19" ry="38" transform="rotate(-46 56 242)"/>
      <ellipse cx="112" cy="326" rx="17" ry="34" transform="rotate(42 112 326)"/>
      <ellipse cx="48" cy="462" rx="18" ry="37" transform="rotate(-40 48 462)"/>
      <ellipse cx="106" cy="566" rx="17" ry="35" transform="rotate(38 106 566)"/>
      <ellipse cx="44" cy="700" rx="19" ry="38" transform="rotate(-42 44 700)"/>
      <ellipse cx="94" cy="804" rx="16" ry="34" transform="rotate(40 94 804)"/>
    </g>
    <path d="M76 118 C118 96 132 72 150 38 M74 278 C110 250 126 220 152 190 M72 518 C116 488 136 448 158 410 M70 742 C104 714 130 678 154 636" fill="none" stroke="currentColor" stroke-width="5" stroke-linecap="round" opacity="0.85"/>
  </svg>
  <main>
    <header>
      <h1>Plant Spectrum Health</h1>
      <p class="subtitle">Upload a with-leaf spectrum photo. The backend applies the fixed crop and ROI from final_model_imgs, calculates absorbance, and estimates plant health.</p>
    </header>

    <section class="panel">
      <form id="uploadForm">
        <label for="leaf">Plant spectrum photo</label>
        <div class="filebox">
          <input id="leaf" name="leaf" type="file" accept="image/*" capture="environment" required>
        </div>
        <p class="hint">Uses the saved no-leaf reference spectrum by default.</p>

        <label for="reference" style="margin-top:14px;">Optional no-leaf reference photo</label>
        <div class="filebox">
          <input id="reference" name="reference" type="file" accept="image/*">
        </div>

        <button type="submit">Analyze Spectrum</button>
      </form>
      <div id="status" class="status">Processing spectrum...</div>
      <div id="error" class="error"></div>
    </section>

    <section id="results" class="results">
      <div class="panel verdict">
        <div id="badge" class="badge">Healthy</div>
        <p id="explanation" class="subtitle"></p>
        <div class="metric-grid">
          <div class="metric"><span>Mean absorbance</span><strong id="meanAbs">0.000</strong></div>
          <div class="metric"><span>Max absorbance</span><strong id="maxAbs">0.000</strong></div>
          <div class="metric"><span>Chlorophyll index</span><strong id="chlIndex">0.000</strong></div>
          <div class="metric"><span>Red absorbance</span><strong id="redAbs">0.000</strong></div>
        </div>
      </div>

      <div class="panel">
        <label>Absorbance graph</label>
        <img id="absorbancePlot" alt="Relative absorbance graph">
      </div>

      <div class="panel">
        <label>ROI used by backend</label>
        <img id="roiPlot" alt="Fixed extraction ROI">
      </div>
    </section>
  </main>

  <script>
    const form = document.getElementById("uploadForm");
    const statusBox = document.getElementById("status");
    const errorBox = document.getElementById("error");
    const results = document.getElementById("results");

    function setMetric(id, value) {
      document.getElementById(id).textContent = Number(value).toFixed(3);
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      statusBox.style.display = "block";
      errorBox.style.display = "none";
      results.style.display = "none";

      try {
        const response = await fetch("/analyze", {
          method: "POST",
          body: new FormData(form)
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Analysis failed.");
        }

        const health = payload.health;
        const badge = document.getElementById("badge");
        badge.textContent = health.verdict;
        badge.className = health.isHealthy ? "badge" : "badge bad";
        document.getElementById("explanation").textContent = health.explanation;
        setMetric("meanAbs", health.meanAbsorbance);
        setMetric("maxAbs", health.maxAbsorbance);
        setMetric("chlIndex", health.chlorophyllIndex);
        setMetric("redAbs", health.redAbsorbance);
        document.getElementById("absorbancePlot").src = "data:image/png;base64," + payload.absorbancePlot;
        document.getElementById("roiPlot").src = "data:image/png;base64," + payload.roiPlot;
        results.style.display = "block";
      } catch (error) {
        errorBox.textContent = error.message;
        errorBox.style.display = "block";
      } finally {
        statusBox.style.display = "none";
      }
    });
  </script>
</body>
</html>"""


class SpectrumHandler(BaseHTTPRequestHandler):
    def send_html(self, body, status=200):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(page_html())
            return
        if parsed.path == "/health":
            self.send_json({"ok": True})
            return
        self.send_html("Not found", status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/analyze":
            self.send_json({"error": "Not found"}, status=404)
            return

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type"),
                },
            )
            if "leaf" not in form or not getattr(form["leaf"], "file", None):
                raise ValueError("Upload a plant spectrum photo first.")

            leaf_bytes = form["leaf"].file.read()
            reference_bytes = None
            if "reference" in form and getattr(form["reference"], "filename", ""):
                reference_bytes = form["reference"].file.read()

            result = analyze_upload(leaf_bytes, reference_bytes)
            self.send_json(result)
        except Exception as exc:
            self.send_json({"error": html.escape(str(exc))}, status=400)


def run():
    parser = argparse.ArgumentParser(description="Run the mobile plant spectrum health web app.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    port = args.port
    server = ThreadingHTTPServer(("127.0.0.1", port), SpectrumHandler)
    print(f"Plant Spectrum Health app running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
