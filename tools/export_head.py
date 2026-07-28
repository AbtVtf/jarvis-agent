"""Export the GNM head + viseme morph targets for the web viewer.

Produces web/assets/head.bin (packed little-endian arrays) and
web/assets/head.json (manifest with offsets/shapes + viseme names).

Layout: template positions (V,3) f32, vertex colors (V,3) u8, triangles
(T,3) u32, then one (V,3) f32 delta block per morph target.
The template is symmetrized so tiny mouth openings don't look crooked.
"""

import json
import os

import numpy as np
from gnm.shape import gnm_numpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "web", "assets")

VISEME_ORDER = ["sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn",
                "RR", "aa", "E", "ih", "oh", "ou", "blink",
                "eyes_up", "eyes_side", "pupil_wide"]

THEMES = {
    "human": {
        "SKIN": [224, 172, 138], "LIP": [204, 122, 108],
        "TEETH": [242, 238, 228], "GUMS": [190, 95, 90],
        "TONGUE": [205, 100, 95], "SCLERA": [245, 243, 240],
        "IRIS": [90, 118, 138], "PUPIL": [18, 18, 20],
        "MOUTH_SOCK": [120, 55, 50],
    },
    # Holographic digital being: cool blues, self-lit eyes.
    "digital": {
        "SKIN": [96, 144, 205], "LIP": [78, 118, 196],
        "TEETH": [205, 232, 248], "GUMS": [62, 92, 165],
        "TONGUE": [84, 120, 190], "SCLERA": [196, 228, 252],
        "IRIS": [64, 205, 255], "PUPIL": [10, 22, 42],
        "MOUTH_SOCK": [26, 42, 84],
    },
}
_T = THEMES[os.environ.get("JARVIS_FACE_THEME", "digital")]
SKIN = np.array(_T["SKIN"])
LIP = np.array(_T["LIP"])
TEETH = np.array(_T["TEETH"])
GUMS = np.array(_T["GUMS"])
TONGUE = np.array(_T["TONGUE"])
SCLERA = np.array(_T["SCLERA"])
IRIS = np.array(_T["IRIS"])
PUPIL = np.array(_T["PUPIL"])
MOUTH_SOCK = np.array(_T["MOUTH_SOCK"])


def vertex_colors(gnm):
  colors = np.tile(SKIN, (gnm.num_vertices, 1)).astype(np.float64)

  def paint(group_names, color):
    for name in group_names if isinstance(group_names, list) else [group_names]:
      w = np.clip(np.array(gnm.vertex_group(name)), 0.0, 1.0)[:, None]
      colors[:] = colors * (1 - w) + color[None, :] * w

  paint("mouth_sock", MOUTH_SOCK)
  paint(["upper_lip", "lower_lip"], LIP)
  paint("gums", GUMS)
  paint("teeth", TEETH)
  paint("tongue", TONGUE)
  paint("scleras", SCLERA)
  paint("eye_sockets", SKIN)
  paint("irises", IRIS)
  paint("pupils", PUPIL)
  return np.clip(colors, 0, 255).astype(np.uint8)


def main():
  gnm = gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD
  )
  mirror = np.array(gnm.mirror_indices)
  sign = np.array([-1.0, 1.0, 1.0])

  tpl = np.array(gnm.template_vertex_positions, dtype=np.float64)
  tpl = 0.5 * (tpl + tpl[mirror] * sign)
  tpl = tpl.astype(np.float32)

  # Drop the transparent cornea layer: opaque rendering would hide the
  # sclera/iris/pupil geometry underneath it.
  tris = np.array(gnm.triangles, dtype=np.uint32)
  cornea = set(gnm.triangle_indices_for_group("eye_exteriors").tolist())
  tris = np.array([t for i, t in enumerate(tris) if i not in cornea],
                  dtype=np.uint32)
  colors = vertex_colors(gnm)

  data = np.load(os.path.join(ROOT, "build", "visemes.npz"))
  morphs = [data[f"delta_{name}"].astype(np.float32) for name in VISEME_ORDER]

  os.makedirs(ASSETS, exist_ok=True)
  manifest = {"numVertices": int(gnm.num_vertices),
              "numTriangles": int(tris.shape[0]),
              "morphNames": VISEME_ORDER,
              "blocks": {}}
  offset = 0
  blobs = []

  def add(name, arr):
    nonlocal offset
    arr = np.ascontiguousarray(arr)
    manifest["blocks"][name] = {
        "offset": offset,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }
    raw = arr.tobytes()
    pad = (-len(raw)) % 4
    blobs.append(raw + b"\0" * pad)
    offset += len(raw) + pad

  add("positions", tpl)
  add("colors", colors)
  add("triangles", tris)
  for name, delta in zip(VISEME_ORDER, morphs):
    add(f"morph_{name}", delta)

  with open(os.path.join(ASSETS, "head.bin"), "wb") as f:
    for b in blobs:
      f.write(b)
  with open(os.path.join(ASSETS, "head.json"), "w") as f:
    json.dump(manifest, f)
  print(f"wrote head.bin ({offset/1e6:.1f} MB), {len(VISEME_ORDER)} morphs")


if __name__ == "__main__":
  main()
