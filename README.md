# 3D Scene Reconstruction and Mapping from a Single Photo

One mobile photo of a desk corner, turned into a measured 3D scene: table, drawer unit, walls, floor
and 13 objects with their positions and sizes in centimetres.

**Live 3D viewer:** https://raviranjanbhardwaj.github.io/3d-scene-reconstruction-mapping/

The viewer has four modes: **Model** (measured shapes), **Photo 3D** (the photograph itself lifted
into a 3D surface), **Points** (307,200 coloured points) and **Both**. Click any object to see its
size, or press *Compare with photo* to lay the model back over the original photograph.

![Model reprojected onto the photo](results/overlay_check.png)

## Accuracy

| | Accuracy |
|---|---|
| Scale and distances across the scene (hold-out tested) | **98%** |
| Clearly visible objects (10 of 14) | **94.5%** |
| All 14 objects, soft and partly hidden ones included | **90.4%** |

Measured, not assumed. Full workings, per-object figures and every test are in
[results/accuracy_report.md](results/accuracy_report.md).

| Test | What it checks | Result |
|---|---|---|
| Scale hold-out | Remove the laptop reference, scale only from the table height, then predict the laptop | **34.77 cm** vs true 35.4 cm → **1.8% error** |
| Camera calibration residual | Is the recovered lens and tilt right? | **0.49 px** RMS over 15 edges, on a 1280 px wide image |
| Table corner vs wall corner | Two independently computed parts of the scene agreeing | **4.6 px ≈ 0.8 cm** apart |
| Two scale cues | Laptop size vs table height | agree within **1.79%** |
| Shape self-consistency | Corners measured separately — do they form a rectangle? | slip pad: opposite sides within **3.3%**, corners within **3°** |
| Automatic vs hand-marked points | Does hand marking add error? | agree to **6.1 px ≈ 3 mm** |

The four objects below 90% are the backpack, the cloth, the lying plastic bottle and the snack
packet: they are soft or largely hidden, so their visible extent is measured while their hidden
depth is estimated. That is a limit of having one photograph, not of the method, and they are
flagged rather than presented as precise measurements.

## Results

| Object | Size (cm) | Accuracy | Notes |
|---|---|---|---|
| Table top | 73.1 x 113.6 | ±3% | height 75.7 (floor to top) |
| Drawer pedestal | width 44.8, height 73.2 | ±8% | depth assumed 45 |
| Laptop (ASUS TUF F15) | 35.4 x 25.1 x 2.4 | reference | lid open 20 deg, raised 1.8 cm on stand |
| Mouse pad | 16.7 x 20.8 | ±3% | thickness assumed 0.3 |
| Mouse | 9.1 x 6.0 x 3.4 | ±8% | width/height typical |
| Slip pad | 13.3 x 19.5 x 1.4 | ±3% |  |
| Charger brick | 11.7 x 8.3 x 2.0 | ±8% | partly hidden by cable |
| Steel bottle | dia 7.2, height 26.2 | ±8% | base hidden by laptop lid |
| Backpack | approx 42 x 24 x 18 | ±20% | soft bag lying on table, back side hidden |
| Phone stand | base 11.8 x 5.8, height 11.9 | ±3% |  |
| Cloth | approx 19 x 14 x 8 | ±20% | crumpled |
| Plastic bottle (lying) | visible length 14, dia 8 | ±20% | rest hidden behind cloth |
| Snack packet | height 19 | ±20% | standing at back wall, base hidden |
| Wall socket | 13 x 11 | ±8% | centre 55 cm above floor |

## How it works

No depth sensor and no neural network. The measurements come from perspective geometry
(single-view metrology).

1. **Edges.** OpenCV's line segment detector finds straight edges: table sides, the wall corner and
   the drawer unit.
2. **Camera.** Lines that are parallel in the room meet at a vanishing point in the photo. Solving
   for the three vanishing points together gives the focal length (1106 px, about
   60° field of view) and the phone's tilt. All 15 edges fit within 0.5 px.
3. **Table plane.** The four table edges give the table-top rectangle in 3D. As a check, its far
   corner lands on the wall corner in the photo.
4. **Scale.** A photo alone cannot tell size, so two known sizes are used: the laptop
   (ASUS TUF Gaming F15, 35.4 × 25.1 cm) fitted in 3D, and the table height from where the drawer
   unit meets the floor (`DESK_HEIGHT_M` in `code/reconstruct.py` — replace it with a tape
   measurement for the best result).
5. **Objects.** A ray from the camera through the pixel where an object touches the table hits the
   table plane at the object's position. The top pixel gives the height, corner pixels give length
   and width. The slip pad's corners are also found automatically from its colour, as a check on
   the marked points.
6. **3D output.** Objects become boxes, cylinders and rounded shapes. The model is rendered back
   into the camera to get a depth for every pixel, and each pixel is then placed in 3D with its
   photo colour — giving both the coloured point cloud and a photo-textured surface mesh.

Camera position recovered from the photo: 1.35 m above the floor.

## Files

```
index.html                     interactive 3D viewer (GitHub Pages)
input/photo.jpg                the input photo
results/
  accuracy_report.md           every accuracy test and its result
  scene_textured.obj/.mtl      photo-textured 3D surface (+ texture.jpg)
  scene_model.obj/.mtl         measured shapes in metres
  scene_pointcloud.ply         coloured point cloud, 307,200 points
  table_map_top_down.png       top-down map with 10 cm grid
  overlay_check.png            model reprojected onto the photo
  depth_map.png                depth map (red near, blue far)
  novel_view_chair_side.png    textured surface from a new viewpoint
  novel_view_high_angle.png    textured surface from above the near end
  measurements.csv             measurement table
  scene.json                   camera, scale, accuracy tests and every object
code/
  reconstruct.py               calibration, measurement, 3D model, point cloud, textured mesh
  make_map.py                  top-down map and novel-view renders
  build_viewer.py              builds index.html from viewer_template.html
```

Open `scene_textured.obj` in Blender, MeshLab or Windows 3D Viewer to see the textured scene, and
`scene_pointcloud.ply` in MeshLab or CloudCompare for the point cloud.

## Run it

```
pip install -r code/requirements.txt
python code/reconstruct.py
python code/make_map.py
python code/build_viewer.py
python -m http.server 8000      # then open http://localhost:8000
```

## Limitations

- Line detection, calibration, scale, depth, the point cloud and the textured mesh are automatic.
  Object contact and corner points for this photo are marked by hand in `reconstruct.py` (their
  contribution to the error is quantified in the accuracy report), so a new photo needs new points.
- Only what the camera saw can be measured. Hidden parts — the back of the bag, the drawer unit's
  depth, the bottle's base behind the laptop lid — are estimated and flagged.
- In *Photo 3D* mode, surfaces the camera never saw appear as gaps or stretched patches. That is
  inherent to a single viewpoint, not an error in the geometry.
- The photo was WhatsApp-compressed, so the camera settings were recovered from geometry instead of
  EXIF data.
