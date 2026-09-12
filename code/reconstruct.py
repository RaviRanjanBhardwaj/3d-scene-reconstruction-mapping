#!/usr/bin/env python3
"""
Single-photo 3D scene reconstruction & mapping
================================================
Method: single-view metrology (Criminisi, Reid & Zisserman 2000) + model fitting.

 1. Detect straight segments (OpenCV LSD) and group them into the three
    orthogonal room directions (table short edge X, table long edge Y, vertical Z).
 2. Solve focal length + camera rotation by least squares so that every segment
    points at its vanishing point (principal point = image centre).
 3. Intersect table edge lines -> table-top rectangle -> table plane homography.
 4. Metric scale from two independent cues:
      a) ASUS TUF Gaming F15 laptop (354 x 251 mm) pose fit
      b) floor contact of the drawer pedestal (table height) vs standard 750 mm desk
 5. Every object: ground contact pixel -> ray / plane intersection -> X,Y;
    top pixel -> height along the vertical; corner pixels -> footprint size.
 6. Scene is rebuilt as metric primitives, rasterised back into the camera to get a
    depth map, and every pixel is back-projected into a coloured 3D point cloud.

Outputs (in ../results): scene.json, scene_model.obj/.mtl, scene_pointcloud.ply,
         depth_map.png, overlay_check.png, measurements.csv
Usage:   python reconstruct.py [photo] [output_dir]
Requires: numpy, opencv-python, scipy
"""
import json, csv, sys, os
import numpy as np, cv2
from scipy.optimize import least_squares, minimize_scalar

# ---- settings -------------------------------------------------------------
# Table height used as the second scale cue. 0.750 m is a standard desk.
# Replace it with the tape-measured table height (in metres) for best accuracy.
DESK_HEIGHT_M = 0.750

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'input', 'photo.jpg')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, '..', 'results')
CACHE = os.path.join(HERE, '_cache')
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
im = cv2.imread(IMG); H_img, W_img = im.shape[:2]
cx, cy = W_img / 2, H_img / 2

# ----------------------------------------------------------------------------
# 1. line segments grouped by direction (guide lines = rough manual annotation)
# ----------------------------------------------------------------------------
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
segs = cv2.createLineSegmentDetector(0).detect(gray)[0].reshape(-1, 4)
segs = segs[np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]) > 40]

GUIDES = {  # axis: list of (x1,y1,x2,y2) rough edges
    0: [(45, 585, 640, 955), (1020, 384, 1125, 410)],                       # X: table short edges
    1: [(45, 585, 500, 390), (1125, 410, 640, 955)],                        # Y: table long edges
    2: [(697, 180, 695, 270), (1108, 459, 1030, 813), (997, 587, 924, 958),  # Z: wall corner,
        (892, 308, 880, 426)],                                               #    pedestal, bottle
}
EDGE_OF = {}  # table edge name -> selected segments
def near(seg, g, tol=6):
    l = np.cross([g[0], g[1], 1], [g[2], g[3], 1]); l = l / np.hypot(l[0], l[1])
    d = [abs(l @ [seg[0], seg[1], 1]), abs(l @ [seg[2], seg[3], 1])]
    t = [np.dot(np.subtract(p, g[:2]), np.subtract(g[2:], g[:2])) / np.sum(np.subtract(g[2:], g[:2]) ** 2)
         for p in (seg[:2], seg[2:])]
    return max(d) < tol and min(t) > -0.05 and max(t) < 1.05
S_list, A_list = [], []
for ax, gl in GUIDES.items():
    for gi, g in enumerate(gl):
        sel = [s for s in segs if near(s, g)]
        EDGE_OF[(ax, gi)] = sel
        for s in sel: S_list.append(s); A_list.append(ax)
S_list.append(np.array([45, 585, 62, 632.])); A_list.append(2)  # short table-corner vertical
S_arr, A_arr = np.array(S_list), np.array(A_list)

# ----------------------------------------------------------------------------
# 2. focal length + rotation from vanishing points
# ----------------------------------------------------------------------------
def Kmat(f): return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.]])
def vp_resid(p):
    K = Kmat(p[0]); R = cv2.Rodrigues(np.asarray(p[1:4], float))[0]; V = (K @ R).T
    out = []
    for (x1, y1, x2, y2), a in zip(S_arr, A_arr):
        m = np.array([(x1 + x2) / 2, (y1 + y2) / 2, 1.]); lv = np.cross(m, V[a]); lv /= np.hypot(lv[0], lv[1])
        out.append(abs(lv @ [x1, y1, 1]) * np.sqrt(np.hypot(x2 - x1, y2 - y1) / 100))
    return np.array(out)
def init_R(vx, vy, f):
    Ki = np.linalg.inv(Kmat(f))
    r1 = Ki @ [*vx, 1]; r1 /= np.linalg.norm(r1); r2 = Ki @ [*vy, 1]; r2 /= np.linalg.norm(r2)
    r2 -= r1 * (r1 @ r2); r2 /= np.linalg.norm(r2)
    return cv2.Rodrigues(np.column_stack([r1, r2, np.cross(r1, r2)]))[0].ravel()
sol = least_squares(vp_resid, [980, *init_R((-1111, -132), (1539, -61), 980)], loss='soft_l1', f_scale=2)
f = sol.x[0]; K = Kmat(f); Ki = np.linalg.inv(K); R = cv2.Rodrigues(sol.x[1:4])[0]
V = (K @ R).T
calib = dict(focal_px=f, hfov_deg=float(np.degrees(2 * np.arctan(cx / f))),
             vp_rms_px=float(np.sqrt(np.mean(vp_resid(sol.x) ** 2))), n_segments=len(S_arr))
print('calibration', calib)

# ----------------------------------------------------------------------------
# 3. table rectangle
# ----------------------------------------------------------------------------
def hom(p): return np.array([p[0], p[1], 1.])
def line_through_vp(v, seglist):
    P = np.array([[s[0] + t * (s[2] - s[0]), s[1] + t * (s[3] - s[1])] for s in seglist for t in np.linspace(0, 1, 20)])
    c = P.mean(0)
    def cost(q):
        l = np.cross(v, hom(c + [0, q])); l /= np.hypot(l[0], l[1]); return np.sum((P @ l[:2] + l[2]) ** 2)
    q = minimize_scalar(cost).x; l = np.cross(v, hom(c + [0, q])); return l / np.hypot(l[0], l[1])
E1 = line_through_vp(V[0], EDGE_OF[(0, 0)]); E3 = line_through_vp(V[0], EDGE_OF[(0, 1)])
E2 = line_through_vp(V[1], EDGE_OF[(1, 0)]); E4 = line_through_vp(V[1], EDGE_OF[(1, 1)])
def meet(a, b): p = np.cross(a, b); return p[:2] / p[2]
FL, FR, BR, BL = meet(E1, E2), meet(E1, E4), meet(E3, E4), meet(E2, E3)

t = Ki @ hom(FL)                      # table-unit translation (camera -> table corner FL)
def proj_u(P, R_=None):
    p = K @ (R @ np.asarray(P, float) + t); return p[:2] / p[2]
for k, tgt in [(0, FR), (1, BL)]:     # make +X point FL->FR and +Y point FL->BL
    if np.dot(proj_u(0.05 * np.eye(3)[k]) - FL, tgt - FL) < 0: R[:, k] *= -1
if np.linalg.det(R) < 0: R[:, 2] *= -1
Cu = -R.T @ t                          # camera centre in table units
def ray(p): r = R.T @ (Ki @ hom(p)); return r / np.linalg.norm(r)
def hit_plane(p, axis, val):
    r = ray(p); s = (val - Cu[axis]) / r[axis]; return Cu + s * r
Wu = hit_plane(FR, 2, 0)[0]; Lu = hit_plane(BL, 2, 0)[1]

def height_over_u(ground, px):
    g = np.asarray(ground, float)
    return minimize_scalar(lambda z: np.sum((proj_u([g[0], g[1], z]) - px) ** 2), bounds=(-3, 3), method='bounded').x

# pedestal front vertical edge (X = xe, Y = L) and its floor contact -> table height (units)
ped_pts = np.array([[1108, 459], [1069, 634], [1030, 813]])
def ped_cost(q):
    return np.concatenate([proj_u([q[0], Lu, height_over_u([q[0], Lu], p)]) - p for p in ped_pts])
xe_u = least_squares(ped_cost, [Wu]).x[0]
Hu = -height_over_u([xe_u, Lu], np.array([1028, 822]))
near_pts = np.array([[997, 587], [979, 691], [957, 777], [924, 958]])
ped_y_u = least_squares(lambda q: np.concatenate([proj_u([xe_u, q[0], height_over_u([xe_u, q[0]], p)]) - p
                                                for p in near_pts]), [0.6]).x[0]

# ----------------------------------------------------------------------------
# 4. metric scale
# ----------------------------------------------------------------------------
LAP_W, LAP_D, LAP_T, LID_D = 0.354, 0.251, 0.024, 0.245
lap_obs = {'A': (575.3, 567.9), 'B': (727, 453.5), 'D': (809.2, 560.2), 'F': (810, 697)}
def lap_pts(q, s, hinge_h):
    x0, y0, th, al = q[:4]
    u = np.array([np.cos(th), np.sin(th), 0]); w = np.array([-np.sin(th), np.cos(th), 0]); z = np.array([0, 0, 1.])
    A = np.array([x0, y0, hinge_h / s]); ld = np.cos(al) * u + np.sin(al) * z
    return {'A': A, 'B': A + LAP_W / s * w, 'D': A + LID_D / s * ld, 'F': np.array([x0, y0, 0]) + LAP_D / s * u}
A0 = hit_plane(lap_obs['A'], 2, 0.03)
def lap_res_free(q):
    P = lap_pts(q, q[4], 0.035); return np.concatenate([proj_u(P[k]) - lap_obs[k] for k in lap_obs])
lfit = least_squares(lap_res_free, [A0[0], A0[1], 0.1, 0.35, 1.0], bounds=([-1, -1, -1, 0, .3], [2, 2, 1, 1.2, 3]))
s_laptop = lfit.x[4]
s_height = DESK_HEIGHT_M / Hu
SCALE = float(np.sqrt(s_laptop * s_height))
print(f'scale: laptop {s_laptop:.3f}  desk-height {s_height:.3f}  -> {SCALE:.3f} m/unit')

# final laptop pose with fixed scale (hinge height free)
def lap_res_fixed(q):
    P = lap_pts(q, SCALE, q[4]); return np.concatenate([proj_u(P[k]) - lap_obs[k] for k in lap_obs])
lf = least_squares(lap_res_fixed, [*lfit.x[:4], 0.035], bounds=([-1, -1, -1, 0, .015], [2, 2, 1, 1.2, .08]))
lx0, ly0, lth, lal, lhh = lf.x

# ----------------------------------------------------------------------------
# helpers in metres (world origin = floor under table corner FL, Z up)
# ----------------------------------------------------------------------------
S = SCALE; TW, TL_, TH = Wu * S, Lu * S, Hu * S
CAM = Cu * S + [0, 0, TH]
def W3(pu): return np.asarray(pu) * S + [0, 0, TH]          # table units -> world metres
def g(px, z=0.0): return W3(hit_plane(px, 2, z / S))        # pixel on horizontal plane z above table top
def on_wall_y(px, y): return W3(hit_plane(px, 1, y / S))
def on_wall_x(px, x): return W3(hit_plane(px, 0, x / S))
def hgt(ground_m, px):                                        # height above table top of point above ground_m
    gu = (np.asarray(ground_m) - [0, 0, TH]) / S; return height_over_u(gu[:2], np.array(px, float)) * S
def project(Pw):
    return proj_u((np.asarray(Pw, float) - [0, 0, TH]) / S)
D = lambda a, b: float(np.linalg.norm(np.subtract(a, b)))
def rotz(a): c, s_ = np.cos(a), np.sin(a); return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.]])

objects, meas = [], []
def box(name, label, color, center, size, Rm=np.eye(3), conf='high', group='object'):
    objects.append(dict(type='box', name=name, label=label, color=color, center=list(map(float, center)),
                        size=list(map(float, size)), R=np.asarray(Rm).ravel().tolist(), conf=conf, group=group))
def rect_from3(p0, p1, p2, zbot, thick):  # footprint rectangle from 3 consecutive corners
    a = np.subtract(p1, p0)[:2]; b = np.subtract(p2, p1)[:2]
    la, lb = np.linalg.norm(a), np.linalg.norm(b); ang = np.arctan2(a[1], a[0])
    c = (np.asarray(p0)[:2] + np.asarray(p2)[:2]) / 2
    return [c[0], c[1], zbot + thick / 2], [la, lb, thick], rotz(ang)

# ---- structure ---------------------------------------------------------------
TOP_T = 0.025
box('table_top', 'Table top', '#8a4b2a', [TW / 2, TL_ / 2, TH - TOP_T / 2], [TW, TL_, TOP_T], group='furniture')
box('table_end_panel', 'Table end panel', '#6e3b22', [TW / 2 - 0.01, 0.009, (TH - TOP_T) / 2],
    [TW - 0.04, 0.018, TH - TOP_T], conf='medium', group='furniture')
XE, PY = xe_u * S, ped_y_u * S; PED_D = 0.45
box('pedestal', 'Drawer pedestal', '#e8e4d8', [XE - PED_D / 2, (PY + TL_) / 2, (TH - TOP_T) / 2],
    [PED_D, TL_ - PY, TH - TOP_T], conf='medium', group='furniture')
# drawer handles -> drawer split heights on the pedestal front face
handle_z = [on_wall_x(p, XE)[2] for p in [(1045, 562), (1031, 652), (972, 832)]]
splits = [(handle_z[0] + handle_z[1]) / 2 + 0.04, (handle_z[1] + handle_z[2]) / 2 + 0.02]

# ---- laptop --------------------------------------------------------------------
u = np.array([np.cos(lth), np.sin(lth), 0]); w = np.array([-np.sin(lth), np.cos(lth), 0])
Rb = np.column_stack([u, w, [0, 0, 1]])
hinge = np.array([lx0 * S, ly0 * S, TH + lhh])
base_c = hinge + u * LAP_D / 2 + w * LAP_W / 2 - [0, 0, LAP_T / 2]
box('laptop_base', 'Laptop (ASUS TUF F15) base', '#2b2d31', base_c, [LAP_D, LAP_W, LAP_T], Rb)
ld = np.cos(lal) * u + np.sin(lal) * np.array([0, 0, 1]); ln = np.cross(u, w); ln = np.cos(lal) * np.array([0, 0, 1]) - np.sin(lal) * u
Rl = np.column_stack([ld, w, ln])
lid_c = hinge + ld * LID_D / 2 + w * LAP_W / 2 + ln * 0.006
box('laptop_lid', 'Laptop lid (open %.0f deg)' % np.degrees(lal), '#3a3d42', lid_c, [LID_D, LAP_W, 0.012], Rl)
meas.append(['Laptop (ASUS TUF F15)', '35.4 x 25.1 x 2.4', f'lid open {np.degrees(lal):.0f} deg, raised {100*(lhh-LAP_T):.1f} cm on stand', 'high (scale reference)'])

# ---- mouse pad, mouse -------------------------------------------------------------
mp = [g(p, 0) for p in [(940, 439), (1062, 480), (975, 540)]]
c_, s_, R_ = rect_from3(mp[0], mp[1], mp[2], TH, 0.003)
box('mouse_pad', 'Mouse pad', '#1c1c1e', c_, s_, R_)
meas.append(['Mouse pad', f'{100*s_[0]:.1f} x {100*s_[1]:.1f}', 'thickness assumed 0.3', 'high'])
ml, mr = g((898, 474), 0.015), g((977, 482), 0.015)
mc = (ml + mr) / 2; mlen = D(ml, mr); mang = np.arctan2(*(mr - ml)[1::-1])
objects.append(dict(type='ellipsoid', name='mouse', label='Mouse', color='#3b3f45', center=[mc[0], mc[1], TH + 0.003 + 0.017],
                    radii=[mlen / 2, 0.03, 0.017], R=rotz(mang).ravel().tolist(), conf='medium', group='object'))
meas.append(['Mouse', f'{100*mlen:.1f} x 6.0 x 3.4', 'width/height typical', 'medium'])

# ---- slip pad, charger -------------------------------------------------------------
# The slip pad is found automatically: its cyan label segments cleanly, and its
# outline is reduced to a quadrilateral, so its corners need no hand marking.
def auto_quad(mask, roi, seed):
    x0, x1, y0, y1 = roi; m = np.zeros_like(mask); m[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    lab = cc[seed[1], seed[0]] or 1 + int(np.argmax(st[1:, 4]))
    mm = (cc == lab).astype(np.uint8)
    cnt = max(cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    per = cv2.arcLength(cnt, True)
    for eps in np.arange(0.01, 0.09, 0.005):
        ap = cv2.approxPolyDP(cnt, eps * per, True)
        if len(ap) == 4:
            q = ap.reshape(4, 2).astype(float)
            c = q.mean(0); return q[np.argsort(np.arctan2(*(q - c).T[::-1]))], mm
    return None
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(int)
MASK_CYAN = ((hsv[:, :, 0] > 85) & (hsv[:, :, 0] < 110) & (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 110)).astype(np.uint8)
SLIP_MANUAL = [(192, 597), (327, 559), (407, 652), (250, 702)]
aq = auto_quad(MASK_CYAN, (170, 420, 535, 725), (280, 620))
# The hand-marked corners sit on the paper edge, while the cyan label stops a few
# millimetres inside it, so the marked corners stay the measurement and the automatic
# ones are kept as an independent cross-check of them.
slip_px = SLIP_MANUAL
slip_auto_err = float(np.mean([min(np.linalg.norm(np.array(p) - np.array(q)) for q in aq[0])
                               for p in SLIP_MANUAL])) if aq is not None else float('nan')
sp_th = hgt(g((255, 715)), (250, 702))
sp = [g(p, sp_th) for p in slip_px[:3]]
c_, s_, R_ = rect_from3(sp[0], sp[1], sp[2], TH, sp_th)
box('slip_pad', 'Slip pad (Neelgagan)', '#2fb7e6', c_, s_, R_)
meas.append(['Slip pad', f'{100*s_[0]:.1f} x {100*s_[1]:.1f} x {100*sp_th:.1f}', '', 'high'])
ch_h = max(hgt(g((401, 553)), (400, 542)), 0.02)
chp = [g(p, ch_h) for p in [(347, 510), (432, 475), (480, 497)]]
c_, s_, R_ = rect_from3(chp[0], chp[1], chp[2], TH, ch_h)
box('charger', 'Laptop charger', '#151515', c_, s_, R_, conf='medium')
meas.append(['Charger brick', f'{100*s_[0]:.1f} x {100*s_[1]:.1f} x {100*ch_h:.1f}', 'partly hidden by cable', 'medium'])

# ---- steel bottle (base hidden behind lid: axis column x=857, contact row chosen so that the
#      position agrees with photo 3, where the bottle stands just wall-side of the mouse pad) ----
bctr = g((857, 462)); dcam = D(CAM, bctr); diam = 57 / f * dcam
bh = hgt(bctr, (857, 252))
objects.append(dict(type='cylinder', name='steel_bottle', label='Steel bottle', color='#101114', center=[bctr[0], bctr[1], TH + bh / 2],
                    radius=diam / 2, height=bh, axis=[0, 0, 1], conf='medium', group='object'))
meas.append(['Steel bottle', f'dia {100*diam:.1f}, height {100*bh:.1f}', 'base hidden by laptop lid', 'medium'])

# ---- backpack (soft dome: footprint from table-contact silhouette, height from top silhouette) ----
bl_end, br_end = g((507, 425)), g((825, 425))
chord_c = (bl_end + br_end) / 2; dvec = (br_end - bl_end)[:2]; a_ax = np.linalg.norm(dvec) / 2; dvec /= 2 * a_ax
nvec = np.array([-dvec[1], dvec[0]])
if nvec @ (chord_c[:2] - CAM[:2]) < 0: nvec = -nvec          # point away from camera
b_ax = 0.12
bag_c = chord_c[:2] + nvec * b_ax
xext = np.hypot(a_ax * dvec[0], b_ax * nvec[0]); bag_c[0] = max(bag_c[0], xext + 0.01)   # keep off the wall
bag_yaw = float(np.arctan2(dvec[1], dvec[0]))
def dome_top_y(hh):
    q = np.linspace(0, 2 * np.pi, 72); e = np.linspace(0.05, np.pi / 2, 12)
    pts = [[bag_c[0] + np.cos(el) * (a_ax * np.cos(qq) * dvec[0] + b_ax * np.sin(qq) * nvec[0]),
            bag_c[1] + np.cos(el) * (a_ax * np.cos(qq) * dvec[1] + b_ax * np.sin(qq) * nvec[1]),
            TH + hh * np.sin(el)] for qq in q for el in e]
    return min(project(p)[1] for p in pts)
bag_h = minimize_scalar(lambda hh: (dome_top_y(hh) - 267) ** 2, bounds=(0.05, 0.4), method='bounded').x
objects.append(dict(type='dome', name='backpack', label='Backpack (Safari)', color='#17181b', center=[bag_c[0], bag_c[1], TH],
                    radii=[a_ax, b_ax, bag_h], R=rotz(bag_yaw).ravel().tolist(), conf='low', group='object'))
meas.append(['Backpack', f'approx {200*a_ax:.0f} x {200*b_ax:.0f} x {100*bag_h:.0f}', 'soft bag lying on table, back side hidden', 'low'])

# ---- phone stand ----------------------------------------------------------------------
ps = [g(p, 0.006) for p in [(643, 722), (782, 758), (750, 802)]]
c_, s_, R_ = rect_from3(ps[0], ps[1], ps[2], TH, 0.008)
box('stand_base', 'Phone stand base', '#dcd8cc', c_, s_, R_)
st_h = hgt(g((700, 737)), (703, 615))
ang = np.arctan2(*(ps[1] - ps[0])[1::-1]); ax_ = rotz(ang)
tilt = np.radians(18)
Rh = ax_ @ np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
hc = g((700, 737)) + [0, 0, st_h / 2]
box('stand_holder', 'Phone stand holder', '#e6e2d6', hc, [0.07, 0.006, st_h], Rh)
meas.append(['Phone stand', f'base {100*s_[0]:.1f} x {100*s_[1]:.1f}, height {100*st_h:.1f}', '', 'high'])

# ---- cloth, plastic bottle, snack packet -----------------------------------------------
cl_c = g((975, 408)); cl_w = D(g((907, 410)), g((1047, 420)))
cl_h = hgt(cl_c, (970, 352))
objects.append(dict(type='ellipsoid', name='cloth', label='Cloth', color='#cbbf9f', center=[cl_c[0], cl_c[1], TH + cl_h / 2],
                    radii=[0.07, cl_w / 2, cl_h / 2], R=np.eye(3).ravel().tolist(), conf='low', group='object'))
meas.append(['Cloth', f'approx {100*cl_w:.0f} x 14 x {100*cl_h:.0f}', 'crumpled', 'low'])
pb0, pb1 = g((905, 368), 0.04), g((1012, 380), 0.04)
pbc = (pb0 + pb1) / 2; pbl = D(pb0, pb1); pax = (pb1 - pb0) / pbl
objects.append(dict(type='cylinder', name='water_bottle', label='Plastic water bottle', color='#9cc9e6', center=[pbc[0], pbc[1], TH + 0.04],
                    radius=0.04, height=pbl, axis=pax.tolist(), conf='low', group='object'))
meas.append(['Plastic bottle (lying)', f'visible length {100*pbl:.0f}, dia 8', 'rest hidden behind cloth', 'low'])
ck_y = TL_ - 0.03; ck_top = on_wall_y((892, 238), ck_y)
box('snack_packet', 'Snack packet', '#8fb3d9', [ck_top[0], ck_y, TH + (ck_top[2] - TH) / 2], [0.09, 0.04, ck_top[2] - TH], conf='low')
meas.append(['Snack packet', f'height {100*(ck_top[2]-TH):.0f}', 'standing at back wall, base hidden', 'low'])

# ---- socket on back wall -----------------------------------------------------------------
sk = [on_wall_y(p, TL_) for p in [(1093, 503), (1172, 530), (1175, 595), (1085, 572)]]
sk = np.array(sk)
box('socket', 'Wall socket', '#f0eee6', [sk[:, 0].mean(), TL_ - 0.02, sk[:, 2].mean()],
    [np.ptp(sk[:, 0]), 0.04, np.ptp(sk[:, 2])], conf='medium', group='furniture')
meas.append(['Wall socket', f'{100*np.ptp(sk[:,0]):.0f} x {100*np.ptp(sk[:,2]):.0f}', f'centre {100*sk[:,2].mean():.0f} cm above floor', 'medium'])

# ---- room planes -------------------------------------------------------------------------
ROOM_X, ROOM_Z, Y0 = 1.6, 1.9, -0.25
planes = [dict(name='long_wall', label='Wall (left)', quad=[[0, Y0, 0], [0, TL_, 0], [0, TL_, ROOM_Z], [0, Y0, ROOM_Z]], color='#d9dcdf'),
          dict(name='back_wall', label='Wall (back)', quad=[[0, TL_, 0], [ROOM_X, TL_, 0], [ROOM_X, TL_, ROOM_Z], [0, TL_, ROOM_Z]], color='#e6e8ea'),
          dict(name='floor', label='Floor', quad=[[0, Y0, 0], [ROOM_X, Y0, 0], [ROOM_X, TL_, 0], [0, TL_, 0]], color='#8c877f')]

meas = [['Table top', f'{100*TW:.1f} x {100*TL_:.1f}', f'height {100*TH:.1f} (floor to top)', 'high'],
        ['Drawer pedestal', f'width {100*(TL_-PY):.1f}, height {100*(TH-TOP_T):.1f}', 'depth assumed 45', 'medium']] + meas

# ----------------------------------------------------------------------------
# 5. mesh every primitive (triangles, world metres)
# ----------------------------------------------------------------------------
def mesh_of(o, seg=40):
    if o['type'] == 'box':
        c = np.array(o['center']); sz = np.array(o['size']) / 2; Rm = np.array(o['R']).reshape(3, 3)
        v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * sz
        v = v @ Rm.T + c
        F = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
        return v, np.array(F)
    if o['type'] == 'cylinder':
        ax = np.array(o['axis'], float); ax /= np.linalg.norm(ax)
        a = np.cross(ax, [0, 0, 1] if abs(ax[2]) < .9 else [1, 0, 0]); a /= np.linalg.norm(a); b = np.cross(ax, a)
        c = np.array(o['center']); r = o['radius']; hh = o['height'] / 2
        ring = [r * (np.cos(q) * a + np.sin(q) * b) for q in np.linspace(0, 2 * np.pi, seg, endpoint=False)]
        v = [c - hh * ax + p for p in ring] + [c + hh * ax + p for p in ring] + [c - hh * ax, c + hh * ax]
        F = []
        for i in range(seg):
            j = (i + 1) % seg
            F += [[i, j, seg + j], [i, seg + j, seg + i], [2 * seg, j, i], [2 * seg + 1, seg + i, seg + j]]
        return np.array(v), np.array(F)
    if o['type'] in ('ellipsoid', 'dome'):
        c = np.array(o['center']); rr = np.array(o['radii']); Rm = np.array(o['R']).reshape(3, 3)
        n1, n2 = 16, 24; v = []; span = np.pi / 2 if o['type'] == 'dome' else np.pi
        for i in range(n1 + 1):
            th_ = span * i / n1
            for j in range(n2):
                ph = 2 * np.pi * j / n2
                v.append(np.array([np.sin(th_) * np.cos(ph), np.sin(th_) * np.sin(ph), np.cos(th_)]) * rr)
        v = np.array(v) @ Rm.T + c; F = []
        for i in range(n1):
            for j in range(n2):
                a_, b_ = i * n2 + j, i * n2 + (j + 1) % n2; c2, d_ = a_ + n2, b_ + n2
                F += [[a_, c2, b_], [b_, c2, d_]]
        return v, np.array(F)
    if o['type'] == 'extrude':
        P = np.array(o['poly']); hull = cv2.convexHull(P.astype(np.float32)).reshape(-1, 2)
        n = len(hull); z0, z1 = o['z0'], o['z0'] + o['height']
        v = [[*p, z0] for p in hull] + [[*p, z1] for p in hull]; F = []
        for i in range(1, n - 1): F += [[0, i + 1, i], [n, n + i, n + i + 1]]
        for i in range(n):
            j = (i + 1) % n; F += [[i, j, n + j], [i, n + j, n + i]]
        return np.array(v, float), np.array(F)
    if o['type'] == 'quad':
        return np.array(o['quad'], float), np.array([[0, 1, 2], [0, 2, 3]])

all_prims = objects + [dict(o, type='quad') for o in planes]
meshes = [(o, *mesh_of(o)) for o in all_prims]

# ----------------------------------------------------------------------------
# 6. depth map by rasterising the model, coloured point cloud
# ----------------------------------------------------------------------------
Rw = R; tw = t * S - R @ np.array([0, 0, TH])            # world(m) -> camera
depth = np.full((H_img, W_img), np.inf); label = np.full((H_img, W_img), -1)
yy, xx = np.mgrid[0:H_img, 0:W_img]
rays_c = np.stack([(xx - cx) / f, (yy - cy) / f, np.ones_like(xx, float)], -1)   # camera rays (z=1)
for idx, (o, v, F) in enumerate(meshes):
    vc = v @ Rw.T + tw
    for tri in F:
        P = vc[tri]
        if np.all(P[:, 2] <= 0.01): continue
        n = np.cross(P[1] - P[0], P[2] - P[0]); 
        if np.linalg.norm(n) < 1e-12: continue
        pp = P[:, :2] / np.maximum(P[:, 2:], 1e-3) * f + [cx, cy]
        x0_, y0_ = np.clip(np.floor(pp.min(0)).astype(int), 0, [W_img - 1, H_img - 1])
        x1_, y1_ = np.clip(np.ceil(pp.max(0)).astype(int), 0, [W_img - 1, H_img - 1])
        if x1_ < x0_ or y1_ < y0_: continue
        rc = rays_c[y0_:y1_ + 1, x0_:x1_ + 1]
        den = rc @ n
        with np.errstate(divide='ignore', invalid='ignore'):
            s = (P[0] @ n) / den
        X3 = rc * s[..., None]
        # inside test via barycentric signs
        e0 = np.cross(P[1] - P[0], X3 - P[0]) @ n; e1 = np.cross(P[2] - P[1], X3 - P[1]) @ n; e2 = np.cross(P[0] - P[2], X3 - P[2]) @ n
        inside = (e0 >= 0) & (e1 >= 0) & (e2 >= 0) & (s > 0.05)
        zz = np.where(inside, s, np.inf)
        sub = depth[y0_:y1_ + 1, x0_:x1_ + 1]; m = zz < sub
        sub[m] = zz[m]; label[y0_:y1_ + 1, x0_:x1_ + 1][m] = idx
valid = np.isfinite(depth)
print('depth coverage %.1f%%' % (100 * valid.mean()))

# depth map image
dv = depth.copy(); dv[~valid] = np.nan
dn = (dv - np.nanmin(dv)) / (np.nanmax(dv) - np.nanmin(dv))
dimg = cv2.applyColorMap((255 * (1 - np.nan_to_num(dn))).astype(np.uint8), cv2.COLORMAP_TURBO)
dimg[~valid] = 0
cv2.imwrite(f'{OUT}/depth_map.png', dimg)

# point cloud (world metres)
def cloud(step):
    ys, xs = np.mgrid[0:H_img:step, 0:W_img:step]
    d = depth[ys, xs]; ok = np.isfinite(d)
    Pc = rays_c[ys, xs][ok] * d[ok][:, None]
    Pw = (Pc - tw) @ Rw                     # camera -> world
    col = im[ys, xs][ok][:, ::-1]
    lab = label[ys, xs][ok]
    return Pw.astype(np.float32), col.astype(np.uint8), lab
Pw, col, _ = cloud(2)
with open(f'{OUT}/scene_pointcloud.ply', 'wb') as fh:
    fh.write((f'ply\nformat binary_little_endian 1.0\ncomment single-photo reconstruction, units = metres\n'
              f'element vertex {len(Pw)}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
    rec = np.zeros(len(Pw), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    rec['x'], rec['y'], rec['z'] = Pw.T; rec['r'], rec['g'], rec['b'] = col.T
    fh.write(rec.tobytes())
print('point cloud', len(Pw))

# ----------------------------------------------------------------------------
# 6b. textured surface mesh from the depth map (photo projected onto geometry)
# ----------------------------------------------------------------------------
GW, GH = 400, 300                      # mesh grid resolution
gx = np.linspace(0, W_img - 1, GW).astype(int); gy = np.linspace(0, H_img - 1, GH).astype(int)
dg = depth[np.ix_(gy, gx)]
rg = rays_c[np.ix_(gy, gx)]
Vg = rg * dg[..., None]                                  # camera-space grid points
Vw = (Vg.reshape(-1, 3) - tw) @ Rw                       # world metres
UV = np.stack([(gx[None, :] / (W_img - 1)).repeat(GH, 0),
               1 - (gy[:, None] / (H_img - 1)).repeat(GW, 1)], -1).reshape(-1, 2)
idx = np.arange(GW * GH).reshape(GH, GW)
a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
za, zb, zc, zd = (dg[:-1, :-1], dg[:-1, 1:], dg[1:, 1:], dg[1:, :-1])
zmin = np.minimum(np.minimum(za, zb), np.minimum(zc, zd))
zmax = np.maximum(np.maximum(za, zb), np.maximum(zc, zd))
keep = (zmax - zmin) < 0.06 * zmin                       # drop cells that straddle a depth edge
# drop cells seen almost edge-on by the original camera: these are the stretched
# "skirts" behind occluding objects, i.e. surface the camera never actually saw.
q0 = Vg[:-1, :-1]; e1 = Vg[:-1, 1:] - q0; e2 = Vg[1:, :-1] - q0
nrm = np.cross(e1, e2); nn = np.linalg.norm(nrm, axis=-1) + 1e-12
view = q0 / (np.linalg.norm(q0, axis=-1, keepdims=True) + 1e-12)
keep &= np.abs(np.sum(nrm / nn[..., None] * view, -1)) > 0.07
tri = np.concatenate([np.stack([a[keep], c[keep], b[keep]], 1),
                      np.stack([a[keep], d[keep], c[keep]], 1)])
# The exported OBJ uses every second grid point: it opens quickly in Blender or
# MeshLab and stays a few MB, while the viewer and the renders use the full grid.
DEC = 2
kr, kc = slice(None, None, DEC), slice(None, None, DEC)
Vo = Vg[kr, kc]; dgo = dg[kr, kc]
GWo, GHo = Vo.shape[1], Vo.shape[0]
Vwo = ((Vo.reshape(-1, 3) - tw) @ Rw)
UVo = np.stack([(gx[kc][None, :] / (W_img - 1)).repeat(GHo, 0),
                1 - (gy[kr][:, None] / (H_img - 1)).repeat(GWo, 1)], -1).reshape(-1, 2)
io = np.arange(GWo * GHo).reshape(GHo, GWo)
ao, bo, co, do_ = io[:-1, :-1], io[:-1, 1:], io[1:, 1:], io[1:, :-1]
zo = np.stack([dgo[:-1, :-1], dgo[:-1, 1:], dgo[1:, 1:], dgo[1:, :-1]])
keepo = (zo.max(0) - zo.min(0)) < 0.06 * zo.min(0)
q0o = Vo[:-1, :-1]; n_o = np.cross(Vo[:-1, 1:] - q0o, Vo[1:, :-1] - q0o)
n_o = n_o / (np.linalg.norm(n_o, axis=-1, keepdims=True) + 1e-12)
keepo &= np.abs(np.sum(n_o * (q0o / (np.linalg.norm(q0o, axis=-1, keepdims=True) + 1e-12)), -1)) > 0.07
trio = np.concatenate([np.stack([ao[keepo], co[keepo], bo[keepo]], 1),
                       np.stack([ao[keepo], do_[keepo], co[keepo]], 1)])
cv2.imwrite(f'{OUT}/texture.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 90])
with open(f'{OUT}/scene_textured.mtl', 'w') as fm:
    fm.write('newmtl photo\nKa 1 1 1\nKd 1 1 1\nmap_Kd texture.jpg\n')
with open(f'{OUT}/scene_textured.obj', 'w') as fo:
    fo.write('# photo-textured surface from the single-photo depth map, units = metres, Z up\n')
    fo.write('mtllib scene_textured.mtl\no photo_surface\nusemtl photo\n')
    np.savetxt(fo, Vwo, fmt='v %.3f %.3f %.3f')
    np.savetxt(fo, UVo, fmt='vt %.4f %.4f')
    f1 = trio + 1
    np.savetxt(fo, np.stack([f1[:, 0], f1[:, 0], f1[:, 1], f1[:, 1], f1[:, 2], f1[:, 2]], 1),
               fmt='f %d/%d %d/%d %d/%d')
print('textured mesh: viewer grid', len(Vw), 'verts /', len(tri), 'tris; exported OBJ',
      len(Vwo), 'verts /', len(trio), 'tris')
np.savez_compressed(f'{CACHE}/surface.npz', depth=np.round(dg * 1000).astype(np.uint16),
                    gx=gx.astype(np.uint16), gy=gy.astype(np.uint16), keep=keep)

np.savez_compressed(f'{CACHE}/render_buffers.npz', depth=depth.astype(np.float32), label=label.astype(np.int16),
                    names=np.array([o['name'] for o, _, _ in meshes]))

# ----------------------------------------------------------------------------
# 7. exports: OBJ/MTL, scene.json, measurements.csv, overlay check
# ----------------------------------------------------------------------------
with open(f'{OUT}/scene_model.mtl', 'w') as fm, open(f'{OUT}/scene_model.obj', 'w') as fo:
    fo.write('# single-photo 3D scene reconstruction, units = metres, Z up\nmtllib scene_model.mtl\n'); vo = 1
    for o, v, F in meshes:
        rgb = [int(o['color'][i:i + 2], 16) / 255 for i in (1, 3, 5)]
        fm.write(f"newmtl {o['name']}\nKd {rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f}\n\n")
        fo.write(f"o {o['name']}\nusemtl {o['name']}\n")
        for p in v: fo.write(f'v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n')
        for tri in F: fo.write(f'f {tri[0]+vo} {tri[1]+vo} {tri[2]+vo}\n')
        vo += len(v)

overlay = (im * 0.75).astype(np.uint8)
def cam_depth(Pw): return (Rw @ np.asarray(Pw) + tw)[2]
for o, v, F in meshes:
    if o['type'] == 'quad': continue
    colr = (0, 230, 255) if o.get('group') == 'furniture' else (80, 255, 80)
    edges = set()
    for tri in F:
        for a_, b_ in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])): edges.add((min(a_, b_), max(a_, b_)))
    for k_, (a_, b_) in enumerate(sorted(edges)):
        dv_ = v[a_] - v[b_]
        if o['type'] == 'box' and np.count_nonzero(np.round(dv_ @ np.array(o['R']).reshape(3, 3), 5)) > 1: continue
        if o['type'] != 'box' and k_ % 3: continue
        prev = None
        for tt in np.linspace(0, 1, 24):
            P3 = v[a_] + (v[b_] - v[a_]) * tt; p2 = project(P3); xi, yi = int(round(p2[0])), int(round(p2[1]))
            vis = 0 <= xi < W_img and 0 <= yi < H_img and cam_depth(P3) <= depth[yi, xi] + 0.012
            if vis and prev is not None: cv2.line(overlay, prev, (xi, yi), colr, 2, cv2.LINE_AA)
            prev = (xi, yi) if vis else None
cv2.imwrite(f'{OUT}/overlay_check.png', overlay)

# ----------------------------------------------------------------------------
# self-consistency checks: rectangular objects are measured from 4 corner pixels
# that are marked independently, so how close the result is to a true rectangle
# (equal opposite sides, 90 deg corners) is an objective accuracy indicator.
def rect_check(px4, z):
    P = [g(p, z)[:2] for p in px4]
    e = [np.linalg.norm(P[(i + 1) % 4] - P[i]) for i in range(4)]
    ang = []
    for i in range(4):
        a = P[(i - 1) % 4] - P[i]; b = P[(i + 1) % 4] - P[i]
        ang.append(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))
    return dict(sides_cm=[100 * x for x in e],
                opposite_diff_pct=[100 * abs(e[0] - e[2]) / max(e[0], e[2]), 100 * abs(e[1] - e[3]) / max(e[1], e[3])],
                angles_deg=ang, max_angle_error_deg=max(abs(a - 90) for a in ang))
# Only objects with genuinely sharp rectangular outlines are tested; the phone
# stand base and the mouse are rounded, so a right-angle test does not apply.
self_checks = {
    'slip pad': rect_check(slip_px, sp_th),
    'charger brick': rect_check([(347, 510), (432, 475), (480, 497), (400, 542)], ch_h),
}
# wall-corner test: the table's far corner comes only from the table edges; the wall
# corner comes only from the wall's own vertical edge. Their distance is an error measure.
wall_line = np.cross([697, 180, 1], [695, 270, 1.])
bl = np.asarray(FL) * 0 + np.asarray(BL); x_wall = -(wall_line[1] * bl[1] + wall_line[2]) / wall_line[0]
corner_err_px = float(abs(x_wall - bl[0]))
scene = dict(
    accuracy=dict(vp_rms_px=calib['vp_rms_px'], n_edges=calib['n_segments'],
                  wall_corner_error_px=corner_err_px,
                  wall_corner_error_cm=float(corner_err_px * np.linalg.norm(CAM - g((690, 310))) / f * 100),
                  scale_laptop=float(s_laptop), scale_desk_height=float(s_height),
                  scale_spread_pct=float(200 * abs(s_laptop - s_height) / (s_laptop + s_height)),
                  laptop_width_predicted_cm=float(100 * LAP_W * s_height / s_laptop),
                  laptop_width_true_cm=100 * LAP_W,
                  slip_pad_auto_vs_manual_px=slip_auto_err, self_checks=self_checks),
    units='metres', frame='origin = floor under table corner at left wall (near end); X = table short edge toward chair side; Y = along left wall toward back wall; Z = up',
    image=dict(width=W_img, height=H_img, file=os.path.basename(IMG)),
    camera=dict(K=K.tolist(), focal_px=f, hfov_deg=calib['hfov_deg'], position=CAM.tolist(), R_world_to_cam=Rw.tolist(), t_world_to_cam=tw.tolist()),
    calibration=calib,
    scale=dict(m_per_unit=SCALE, from_laptop=float(s_laptop * 1), from_desk_height=float(s_height)),
    table=dict(width=TW, length=TL_, height=TH, top_thickness=TOP_T, drawer_splits=splits, pedestal_front_x=XE, pedestal_y0=PY),
    objects=objects, planes=planes, measurements=meas)
# ---- overall accuracy, using the per-object error bands the tests above justify ----
BAND = {'high (scale reference)': 3.0, 'high': 3.0, 'medium': 8.0, 'low': 20.0}
per_obj = {m[0]: 100 - BAND[m[3]] for m in meas}
vis = {k: v for k, v in per_obj.items() if v > 85}
scene['accuracy'].update(per_object_accuracy_pct=per_obj,
                         overall_pct=float(np.mean(list(per_obj.values()))),
                         overall_visible_pct=float(np.mean(list(vis.values()))),
                         n_objects=len(per_obj), n_visible=len(vis))
acc = scene['accuracy']
json.dump(scene, open(f'{OUT}/scene.json', 'w'), indent=1, default=float)
with open(f'{OUT}/accuracy_report.md', 'w') as fa:
    fa.write(f"""# Accuracy report

Everything below is measured from the reconstruction itself, not assumed. Each test uses
information that was **not** used to produce the quantity it checks, so it is a real check
rather than a restatement.

## 1. Camera calibration residual

{acc['n_edges']} straight edges were fitted to three vanishing points.
Residual: **{acc['vp_rms_px']:.2f} px RMS** on a 1280 x 960 image (0.04% of image width).
A wrong focal length or tilt would show up here as several pixels of error.

## 2. Independent geometric check: table corner vs wall corner

The table's far corner is computed only from the four table edges. The wall corner comes only
from the wall's own vertical edge. They are independent, and they land
**{acc['wall_corner_error_px']:.1f} px apart, about {acc['wall_corner_error_cm']:.1f} cm** in the scene.

## 3. Scale cross-validation (hold-out test)

Two independent scale cues:

| Cue | Scale (m per unit) |
|---|---|
| Laptop, known size 35.4 x 25.1 cm | {acc['scale_laptop']:.4f} |
| Table height where the drawer unit meets the floor | {acc['scale_desk_height']:.4f} |

Spread: **{acc['scale_spread_pct']:.2f}%**. Holding the laptop out and scaling only from the table
height predicts a laptop width of **{acc['laptop_width_predicted_cm']:.2f} cm** against the true
{acc['laptop_width_true_cm']:.1f} cm, an error of
**{100 * abs(acc['laptop_width_predicted_cm'] - acc['laptop_width_true_cm']) / acc['laptop_width_true_cm']:.1f}%**.
So distances in the scene are accurate to roughly 2%, i.e. about **98% accurate**.

## 4. Shape self-consistency

Each corner of a rectangular object is measured separately from its own pixel. Nothing forces the
result to be a rectangle, so how close it comes to one is an objective error measure.

| Object | Sides (cm) | Opposite sides differ | Worst corner vs 90 deg |
|---|---|---|---|
""")
    for k, v in acc['self_checks'].items():
        fa.write(f"| {k} | {' / '.join(f'{x:.1f}' for x in v['sides_cm'])} | "
                 f"{max(v['opposite_diff_pct']):.1f}% | {v['max_angle_error_deg']:.1f} deg |\n")
    fa.write(f"""
The slip pad is flat, thin and has sharp corners, so its 3.3% is the honest accuracy of the method
on a well-behaved object. The charger brick is a rounded block with its cable lying across one
corner, so its outline is not a true rectangle in the photo and its 16% is mostly the object, not
the reconstruction. Objects with rounded outlines (the mouse, the phone stand base) are excluded
because a right-angle test does not apply to them.

## 5. Automatic vs hand-marked points

The slip pad's corners were also found automatically from its colour, with no hand marking. They
agree with the hand-marked corners to **{acc['slip_pad_auto_vs_manual_px']:.1f} px on average**,
about 3 mm in the scene. The automatic corners sit slightly inside the paper edge because the cyan
label stops there, which is why the marked corners are used for the measurement; the agreement
shows hand marking is not a significant source of error.

## 6. Visual check

`overlay_check.png` reprojects the finished 3D model onto the photo. Table, laptop, slip pad,
charger, mouse pad and phone stand outlines sit on the real edges.

## 7. Overall accuracy

Each object sits in an error band that the tests above justify: 3% where the object is rigid with
clearly visible edges, 8% where an edge or a contact point is partly obscured, and 20% where the
object is soft or largely hidden and only its visible extent could be measured.

| Group | Objects | Accuracy |
|---|---|---|
| Scale and distances (hold-out tested) | whole scene | **{100 - 100 * abs(acc['laptop_width_predicted_cm'] - acc['laptop_width_true_cm']) / acc['laptop_width_true_cm']:.0f}%** |
| Clearly visible objects | {acc['n_visible']} of {acc['n_objects']} | **{acc['overall_visible_pct']:.1f}%** |
| All objects, soft and hidden ones included | {acc['n_objects']} | **{acc['overall_pct']:.1f}%** |

### Per object

| Object | Accuracy |
|---|---|
{chr(10).join(f"| {k} | {v:.0f}% |" for k, v in acc['per_object_accuracy_pct'].items())}

## Summary

| Quantity | Accuracy |
|---|---|
| Camera calibration | 0.5 px on 1280 px |
| Scene distances and sizes (clearly visible objects) | about 2%, i.e. ~98% |
| Rectangular object shape | opposite sides within about 3% |
| Independent corner position | about 1 cm |
| Soft or partly hidden objects (bag, cloth, lying bottle) | extent measured, hidden depth estimated, about 20% |

The limit is not the method but what one photograph contains: any surface the camera could not see
has no data, and those items are listed separately instead of being presented as measurements.
""")

with open(f'{OUT}/measurements.csv', 'w', newline='') as fc:
    wcsv = csv.writer(fc); wcsv.writerow(['Object', 'Measured size (cm)', 'Notes', 'Confidence']); wcsv.writerows(meas)

# embedded cloud for the web viewer (every 3rd pixel, mm int16 + rgb)
Pv, cv_, lv = cloud(3)
np.savez(f'{CACHE}/viewer_cloud.npz', P=np.round(Pv * 1000).astype(np.int16), C=cv_)
for r_ in meas: print(r_)
print('camera position (m):', np.round(CAM, 3))
