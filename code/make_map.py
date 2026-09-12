import numpy as np, cv2, json, os
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
S=json.load(open(os.path.join(RES, 'scene.json'))); T=S['table']
ppm=900; mL=170; mT=110
W=int(T['length']*ppm)+mL+110; H=int(T['width']*ppm)+2*mT+60
img=np.full((H,W,3),(246,244,240),np.uint8)
def uv(x,y): return (int(mL+y*ppm), int(mT+x*ppm))
for k in range(0,12): cv2.line(img,uv(0,k*0.1),uv(T['width'],k*0.1),(226,222,215),1)
for k in range(0,8):
    if k*0.1<=T['width']: cv2.line(img,uv(k*0.1,0),uv(k*0.1,T['length']),(226,222,215),1)
cv2.rectangle(img,(mL-14,mT-14),uv(-0.002,T['length']+0.018),(150,150,150),-1)
cv2.rectangle(img,uv(-0.015,T['length']+0.002),uv(T['width']+0.05,T['length']+0.018),(150,150,150),-1)
cv2.putText(img,'left wall',(mL,mT-22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(90,90,90),1,cv2.LINE_AA)
p=uv(-0.025,T['length']-0.12); cv2.putText(img,'back wall',p,cv2.FONT_HERSHEY_SIMPLEX,0.55,(90,90,90),1,cv2.LINE_AA)
cv2.rectangle(img,uv(0,0),uv(T['width'],T['length']),(42,75,138),3)
x0,y0,xf=T['pedestal_front_x']-0.45,T['pedestal_y0'],T['pedestal_front_x']
def dashed(a,b,n=24):
    a,b=np.array(a),np.array(b)
    for i in range(0,n,2): cv2.line(img,uv(*(a+(b-a)*i/n)),uv(*(a+(b-a)*(i+1)/n)),(120,120,120),1,cv2.LINE_AA)
dashed((x0,y0),(xf,y0)); dashed((xf,y0),(xf,T['length'])); dashed((x0,y0),(x0,T['length']))
q=uv(x0+0.03,y0+0.008); cv2.putText(img,'drawer unit below',q,cv2.FONT_HERSHEY_SIMPLEX,0.45,(110,110,110),1,cv2.LINE_AA)
colors={'laptop_base':(60,55,50),'laptop_lid':(90,85,80),'mouse_pad':(40,40,40),'mouse':(110,100,95),'slip_pad':(230,183,47),
        'charger':(30,30,30),'steel_bottle':(25,25,25),'backpack':(70,65,60),'stand_base':(190,195,200),'stand_holder':(160,165,170),
        'cloth':(159,191,203),'water_bottle':(230,201,156),'snack_packet':(217,179,143)}
names={'laptop_base':('Laptop',0),'mouse_pad':('Mouse pad',0.055),'mouse':('Mouse',-0.01),'slip_pad':('Slip pad',0),'charger':('Charger',0),
       'steel_bottle':('Bottle',0),'backpack':('Backpack',0),'stand_base':('Phone stand',0),'cloth':('Cloth',0.03),'water_bottle':('Plastic bottle',-0.02),'snack_packet':('Snack',0)}
def footprint(o):
    if o['type']=='box':
        c=np.array(o['center']); R=np.array(o['R']).reshape(3,3); h=np.array(o['size'])/2
        pts=[c+R@(h*np.array([sx,sy,sz])) for sx in(-1,1) for sy in(-1,1) for sz in(-1,1)]
    elif o['type']=='cylinder':
        a=np.array(o['axis']); c=np.array(o['center']); e=np.cross(a,[0,0,1]) if abs(a[2])<.9 else np.array([1.,0,0]); e/=np.linalg.norm(e); e2=np.cross(a,e)
        pts=[c+s*a*o['height']/2+o['radius']*(np.cos(q)*e+np.sin(q)*e2) for q in np.linspace(0,2*np.pi,40) for s in(-1,1)]
    else:
        c=np.array(o['center']); R=np.array(o['R']).reshape(3,3); r=o['radii']
        pts=[c+R@np.array([r[0]*np.cos(q),r[1]*np.sin(q),0]) for q in np.linspace(0,2*np.pi,48)]
    return cv2.convexHull(np.array([uv(p[0],p[1]) for p in pts],np.int32))
objs={o['name']:o for o in S['objects']}
for n in ['slip_pad','charger','backpack','mouse_pad','laptop_base','laptop_lid','mouse','stand_base','stand_holder','steel_bottle','cloth','water_bottle','snack_packet']:
    hull=footprint(objs[n]); a=0.25 if n=='laptop_lid' else 0.55
    ov=img.copy(); cv2.fillPoly(ov,[hull],colors[n]); img=cv2.addWeighted(ov,a,img,1-a,0)
    cv2.polylines(img,[hull],True,(30,30,30),1,cv2.LINE_AA)
for n,(lab,dx) in names.items():
    c=objs[n]['center']; p=uv(c[0]+dx,c[1])
    (tw,th),_=cv2.getTextSize(lab,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
    cv2.rectangle(img,(p[0]-tw//2-4,p[1]-th-4),(p[0]+tw//2+4,p[1]+5),(255,255,255),-1)
    cv2.putText(img,lab,(p[0]-tw//2,p[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(20,20,20),1,cv2.LINE_AA)
blue=(0,120,210); yb=T['width']+0.08
for e in (0,T['length']): cv2.arrowedLine(img,uv(yb,T['length']/2),uv(yb,e),blue,2,tipLength=0.02)
t=f"{T['length']*100:.1f} cm"; (tw,_),_=cv2.getTextSize(t,cv2.FONT_HERSHEY_SIMPLEX,0.65,2); p=uv(yb+0.04,T['length']/2)
cv2.putText(img,t,(p[0]-tw//2,p[1]),cv2.FONT_HERSHEY_SIMPLEX,0.65,blue,2,cv2.LINE_AA)
xl=-0.13
for e in (0,T['width']): cv2.arrowedLine(img,uv(T['width']/2,xl),uv(e,xl),blue,2,tipLength=0.03)
p=uv(T['width']/2,xl); cv2.putText(img,f"{T['width']*100:.1f} cm",(p[0]+8,p[1]+6),cv2.FONT_HERSHEY_SIMPLEX,0.6,blue,2,cv2.LINE_AA)
cv2.putText(img,f"Top-down map, 10 cm grid. Table height {T['height']*100:.1f} cm. Chair side at the bottom.",(mL,H-24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(40,40,40),1,cv2.LINE_AA)
cv2.imwrite(os.path.join(RES, 'table_map_top_down.png'), img)

# ---- photo-textured surface rendered from new viewpoints -------------------
HERE = os.path.dirname(os.path.abspath(__file__))
sf = np.load(os.path.join(HERE, '_cache', 'surface.npz'))
dg = sf['depth'].astype(float) / 1000.0; gx = sf['gx']; gy = sf['gy']; keep = sf['keep']
photo = cv2.imread(os.path.join(HERE, '..', 'input', 'photo.jpg'))
cam = S['camera']; K = np.array(cam['K']); Rw = np.array(cam['R_world_to_cam']); tw = np.array(cam['t_world_to_cam'])
GH, GW = dg.shape
uu, vv = np.meshgrid(gx.astype(float), gy.astype(float))
rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
Vw = ((rays * dg[..., None]).reshape(-1, 3) - tw) @ Rw          # world metres
col = photo[np.ix_(gy, gx)].reshape(-1, 3)

def render_surface(eye, target, name, title, Wd=1200, Hd=850, f=950):
    eye = np.array(eye, float); z = np.array(target, float) - eye; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1]); x /= np.linalg.norm(x); y = np.cross(z, x)
    P = (Vw - eye) @ np.stack([x, y, z]).T
    with np.errstate(divide='ignore', invalid='ignore'):
        u = P[:, 0] / P[:, 2] * f + Wd / 2; v = P[:, 1] / P[:, 2] * f + Hd / 2
    uv = np.stack([u, v], 1).reshape(GH, GW, 2); zz = P[:, 2].reshape(GH, GW)
    img = np.full((Hd, Wd, 3), (52, 46, 42), np.uint8)
    q = np.argwhere(keep)                                        # quad top-left indices
    zq = zz[q[:, 0], q[:, 1]]
    order = np.argsort(-zq)                                      # far to near (painter)
    cq = col.reshape(GH, GW, 3)
    for i in order:
        r, c = q[i]
        if min(zz[r, c], zz[r, c + 1], zz[r + 1, c + 1], zz[r + 1, c]) <= 0.05: continue
        poly = np.array([uv[r, c], uv[r, c + 1], uv[r + 1, c + 1], uv[r + 1, c]])
        if not np.isfinite(poly).all(): continue
        if poly[:, 0].max() < 0 or poly[:, 0].min() > Wd or poly[:, 1].max() < 0 or poly[:, 1].min() > Hd: continue
        shade = cq[r:r + 2, c:c + 2].reshape(-1, 3).mean(0)
        cv2.fillConvexPoly(img, np.round(poly).astype(np.int32), shade.tolist(), cv2.LINE_8)
    cv2.putText(img, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(img, 'photo-textured 3D surface; dark areas were hidden from the camera',
                (18, Hd - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (205, 205, 205), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(RES, name), img)

c = [T['width'] / 2, T['length'] / 2, T['height']]
render_surface([T['width'] + 1.05, 0.30, T['height'] + 0.50], [c[0] - 0.05, c[1], c[2]],
               'novel_view_chair_side.png', 'New view from the chair side')
render_surface([c[0] + 0.30, c[1] - 0.75, T['height'] + 0.95], [c[0], c[1] + 0.05, T['height']],
               'novel_view_high_angle.png', 'New view from above the near end')
