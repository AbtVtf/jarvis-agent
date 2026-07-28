"""Author viseme poses in GNM expression space.

Derives primitive articulation directions (jaw open, lip widen, pucker,
lip press, lower-lip raise, tongue up, blink) as minimal-norm expression
coefficient vectors that satisfy landmark-based geometric constraints,
then composes the standard Oculus 15-viseme set from them.

Outputs:
  build/visemes.npz   coefficient vectors (383,) per viseme + blink
  build/visemes_sheet.png   labeled contact-sheet render for inspection
"""

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import imageio
import numpy as np
import pyrender
import trimesh
from gnm.shape import gnm_landmarks
from gnm.shape import gnm_numpy
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")

# Landmark indices (iBUG-68, 0-indexed).
LM_MOUTH_L, LM_MOUTH_R = 48, 54
LM_TOP_OUT, LM_BOT_OUT = 51, 57
LM_TOP_IN, LM_BOT_IN = 62, 66
LM_OUTER_MOUTH = list(range(48, 60))
LM_LOWER_LIP_OUT = [56, 57, 58]
LM_LEYE_PAIRS = [(37, 41), (38, 40)]
LM_REYE_PAIRS = [(43, 47), (44, 46)]

X, Y, Z = 0, 1, 2


def main():
  gnm = gnm_numpy.GNM.from_local(
      version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD
  )
  expr_basis = np.array(gnm.expression_basis)  # (E, V, 3)
  names = list(gnm.expression_names)
  edim = gnm.expression_dim

  idx_mouth = np.array([i for i, n in enumerate(names) if n.startswith("lower_face")])
  idx_tongue = np.array([i for i, n in enumerate(names) if n.startswith("tongue")])
  idx_eyes = np.array([i for i, n in enumerate(names) if "eye_region" in n])

  # Linear map from expression coeffs to landmark deltas: (E, 68, 3).
  cfg = gnm_landmarks.load_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)
  lm_basis = np.einsum("elkd,lk->eld", expr_basis[:, cfg.indices, :], cfg.weights)

  _, lm0 = gnm.vertices_and_landmarks(gnm_landmarks.GNMLandmarksType.HEAD_SPARSE_68)

  def grad(pairs):
    """Gradient of sum of (sign * lm[i][axis]) terms; pairs = [(i, axis, sign)]."""
    c = np.zeros(edim)
    for i, axis, sign in pairs:
      c += sign * lm_basis[:, i, axis]
    return c

  def solve(constraints, allowed):
    """Minimal-norm x (restricted to `allowed` indices) with c_i . x = b_i."""
    rows = np.stack([c for c, _ in constraints])
    b = np.array([t for _, t in constraints])
    sub = rows[:, allowed]
    x_sub = sub.T @ np.linalg.solve(sub @ sub.T, b)
    x = np.zeros(edim)
    x[allowed] = x_sub
    return x

  c_gap_in = grad([(LM_TOP_IN, Y, 1), (LM_BOT_IN, Y, -1)])
  c_gap_out = grad([(LM_TOP_OUT, Y, 1), (LM_BOT_OUT, Y, -1)])
  c_width = grad([(LM_MOUTH_R, X, 1), (LM_MOUTH_L, X, -1)])
  c_protr = grad([(i, Z, 1.0 / len(LM_OUTER_MOUTH)) for i in LM_OUTER_MOUTH])
  c_llip_y = grad([(i, Y, 1.0 / len(LM_LOWER_LIP_OUT)) for i in LM_LOWER_LIP_OUT])

  # Tongue tip = forward-most tongue vertices.
  tongue_v = gnm.vertex_group_indices("tongue")
  tpl = np.array(gnm.template_vertex_positions)
  tip_order = np.argsort(tpl[tongue_v, Z])[::-1]
  tip_idx = tongue_v[tip_order[:30]]
  c_tongue_y = expr_basis[:, tip_idx, Y].mean(axis=1)
  c_tongue_z = expr_basis[:, tip_idx, Z].mean(axis=1)

  c_blink = np.zeros(edim)
  for a, b_ in LM_LEYE_PAIRS + LM_REYE_PAIRS:
    c_blink += lm_basis[:, a, Y] - lm_basis[:, b_, Y]
  eye_gap_total = sum(
      lm0[a, Y] - lm0[b_, Y] for a, b_ in LM_LEYE_PAIRS + LM_REYE_PAIRS
  )

  # Primitive directions: unit = the listed geometric deltas (meters).
  prim = {
      "JAW": solve([(c_gap_in, 0.022)], idx_mouth),
      "WIDE": solve([(c_width, 0.012), (c_gap_in, 0.0)], idx_mouth),
      "PUCK": solve([(c_width, -0.016), (c_protr, 0.006)], idx_mouth),
      "PRESS": solve([(c_gap_out, -0.004), (c_gap_in, -0.003)], idx_mouth),
      "LLUP": solve([(c_llip_y, 0.005)], idx_mouth),
      "TNGU": solve([(c_tongue_y, 0.008), (c_tongue_z, 0.004)], idx_tongue),
      "BLNK": solve([(c_blink, -(eye_gap_total - 0.002))], idx_eyes),
  }

  recipes = {
      "sil": {},
      "PP": {"PRESS": 0.7},
      "FF": {"LLUP": 1.0, "JAW": 0.1},
      "TH": {"JAW": 0.25, "TNGU": 0.8},
      "DD": {"JAW": 0.2, "TNGU": 0.6},
      "kk": {"JAW": 0.3, "WIDE": 0.1},
      "CH": {"JAW": 0.25, "PUCK": 0.6},
      "SS": {"JAW": 0.12, "WIDE": 0.4},
      "nn": {"JAW": 0.18, "TNGU": 0.6},
      "RR": {"JAW": 0.2, "PUCK": 0.4, "TNGU": 0.3},
      "aa": {"JAW": 1.0},
      "E": {"JAW": 0.45, "WIDE": 0.4},
      "ih": {"JAW": 0.3, "WIDE": 0.2},
      "oh": {"JAW": 0.6, "PUCK": 0.7},
      "ou": {"JAW": 0.3, "PUCK": 1.0},
      "blink": {"BLNK": 1.0},
  }

  # Vertex-space deltas, symmetrized across the sagittal plane.
  tpl = np.array(gnm.template_vertex_positions)
  mirror = np.array(gnm.mirror_indices)
  mirror_sign = np.array([-1.0, 1.0, 1.0])

  visemes = {}
  deltas = {}
  for name, recipe in recipes.items():
    x = np.zeros(edim)
    for p, w in recipe.items():
      x += w * prim[p]
    visemes[name] = x
    delta = np.einsum("e,evd->vd", x, expr_basis)
    delta = 0.5 * (delta + delta[mirror] * mirror_sign)
    deltas[name] = delta.astype(np.float32)
    if np.abs(x).max() > 0:
      print(f"{name:6s} |x|max={np.abs(x).max():.2f} "
            f"peak_delta={np.abs(delta).max() * 1000:.1f}mm")

  # Gaze + pupil morphs for the "accessing the mainframe" thinking idle.
  # Eyeball joint rotations baked as morphs (linear approx, fine for <30 deg).
  jn = list(gnm.joint_names)
  eye_joints = [jn.index("left_eye"), jn.index("right_eye")]

  def joint_delta(rotvec):
    rot = np.zeros((gnm.num_joints, 3))
    for j in eye_joints:
      rot[j] = rotvec
    return (np.array(gnm(rotations=rot)) - tpl).astype(np.float32)

  deltas["eyes_up"] = joint_delta([-0.5, 0.0, 0.0])
  deltas["eyes_side"] = joint_delta([0.0, 0.3, 0.0])
  pupil_i = next(i for i, n in enumerate(names) if n.startswith("pupils"))
  deltas["pupil_wide"] = (2.5 * expr_basis[pupil_i]).astype(np.float32)

  os.makedirs(BUILD, exist_ok=True)
  np.savez(
      os.path.join(BUILD, "visemes.npz"),
      **{f"vis_{k}": v for k, v in visemes.items()},
      **{f"delta_{k}": v for k, v in deltas.items()},
  )

  render_sheet(gnm, deltas)


def render_sheet(gnm, deltas):
  faces = np.array(gnm.triangles)
  tpl = np.array(gnm.template_vertex_positions)
  scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.12, 1.0], ambient_light=[0.3] * 3)
  cam = pyrender.PerspectiveCamera(yfov=0.30)
  cam_pose = np.eye(4)
  cam_pose[:3, 3] = [0, 0.235, 0.62]
  scene.add(cam, pose=cam_pose)
  light_pose = np.eye(4)
  light_pose[:3, :3] = Rotation.from_euler("xy", [-15, 25], degrees=True).as_matrix()
  scene.add(pyrender.DirectionalLight(intensity=3.0), pose=light_pose)

  renderer = pyrender.OffscreenRenderer(400, 400)
  tiles = []
  for name, delta in deltas.items():
    tm = trimesh.Trimesh(vertices=tpl + delta, faces=faces, process=False)
    mesh_node = scene.add(pyrender.Mesh.from_trimesh(tm, smooth=True))
    color, _ = renderer.render(scene)
    scene.remove_node(mesh_node)
    tile = color.copy()
    cv2.putText(tile, name, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 220, 80), 2, cv2.LINE_AA)
    tiles.append(tile)

  cols = 4
  rows = (len(tiles) + cols - 1) // cols
  while len(tiles) < rows * cols:
    tiles.append(np.zeros_like(tiles[0]))
  sheet = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
  imageio.imwrite(os.path.join(BUILD, "visemes_sheet.png"), sheet)
  print("wrote", os.path.join(BUILD, "visemes_sheet.png"))


if __name__ == "__main__":
  main()
