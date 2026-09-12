import json, base64, os, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results')
scene = json.load(open(f'{OUT}/scene.json'))
cl = np.load(os.path.join(HERE, '_cache', 'viewer_cloud.npz'))
sf = np.load(os.path.join(HERE, '_cache', 'surface.npz'))
P = cl['P'].astype('<i2'); C = cl['C'].astype(np.uint8)

def jpg_b64(path, q=82, maxw=None):
    im = cv2.imread(path)
    if maxw and im.shape[1] > maxw:
        im = cv2.resize(im, (maxw, int(im.shape[0] * maxw / im.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, q]); return base64.b64encode(buf).decode()

keep = sf['keep']
data = dict(scene=scene, n=int(len(P)),
            surf=dict(gw=int(len(sf['gx'])), gh=int(len(sf['gy']))))
rows = {
    'Table top': ['table_top'], 'Drawer pedestal': ['pedestal'], 'Laptop (ASUS TUF F15)': ['laptop_base', 'laptop_lid'],
    'Mouse pad': ['mouse_pad'], 'Mouse': ['mouse'], 'Slip pad': ['slip_pad'], 'Charger brick': ['charger'],
    'Steel bottle': ['steel_bottle'], 'Backpack': ['backpack'], 'Phone stand': ['stand_base', 'stand_holder'],
    'Cloth': ['cloth'], 'Plastic bottle (lying)': ['water_bottle'], 'Snack packet': ['snack_packet'], 'Wall socket': ['socket']}
data['rows'] = rows

html = open(os.path.join(HERE, 'viewer_template.html')).read()
html = (html.replace('/*__DATA__*/', json.dumps(data, default=float))
            .replace('__PTS__', base64.b64encode(P.tobytes()).decode())
            .replace('__COLS__', base64.b64encode(C.tobytes()).decode())
            .replace('__PHOTO__', jpg_b64(os.path.join(HERE, '..', 'input', 'photo.jpg'), 90))
            .replace('__OVERLAY__', jpg_b64(f'{OUT}/overlay_check.png', 80, 1100))
            .replace('__DEPTH__', jpg_b64(f'{OUT}/depth_map.png', 80, 1100))
            .replace('__SDEPTH__', base64.b64encode(sf['depth'].astype('<u2').tobytes()).decode())
            .replace('__SGX__', base64.b64encode(sf['gx'].astype('<u2').tobytes()).decode())
            .replace('__SGY__', base64.b64encode(sf['gy'].astype('<u2').tobytes()).decode())
            .replace('__SKEEP__', base64.b64encode(np.packbits(keep.ravel()).tobytes()).decode()))
open(os.path.join(HERE, '..', 'index.html'), 'w').write(html)
print('html bytes', len(html))
