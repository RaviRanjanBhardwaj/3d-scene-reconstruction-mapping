# 3D Scene Reconstruction and Mapping from a Single Photo

One mobile photo of my desk corner, turned into a measured 3D scene: table, drawer unit, walls, floor and 13 objects with their positions and sizes.

**Live 3D viewer:** https://raviranjanbhardwaj.github.io/3d-scene-reconstruction-mapping/

![Model drawn back onto the photo](results/overlay_check.png)

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

No depth sensor and no neural network. The measurements come from perspective geometry (single-view metrology).

1. **Edges.** OpenCV's line segment detector finds straight edges: table sides, the wall corner and the drawer unit.
2. **Camera.** Lines that are parallel in the room meet at a vanishing point in the photo. Solving for the three vanishing points together gives the focal length (1106 px, about 60° field of view) and the phone's tilt. All 15 edges fit within 0.5 px.
3. **Table plane.** The four table edges give the table-top rectangle in 3D. As a check, its back corner lands on the wall corner in the photo.
4. **Scale.** A photo alone can't tell size, so two known sizes are used:
   - the laptop (ASUS TUF Gaming F15, 35.4 × 25.1 cm), fitted in 3D
   - the table height, from where the drawer unit meets the floor (`DESK_HEIGHT_M` in `code/reconstruct.py`)

   The two estimates agree within 1.8%.
5. **Objects.** A ray from the camera through the pixel where an object touches the table hits the table plane at the object's position. The top pixel gives the height. Corner pixels give length and width.
6. **3D output.** Objects are modelled as boxes, cylinders and rounded shapes. The model is rendered back into the camera to get a depth map, and every pixel is placed in 3D with its colour to make the point cloud.

Camera position recovered from the photo: 1.35 m above the floor.

## Files

```
index.html                 interactive 3D viewer (GitHub Pages)
input/photo.jpg            the input photo
results/
  scene_model.obj/.mtl     3D model in metres (Blender, MeshLab, Windows 3D Viewer)
  scene_pointcloud.ply     coloured point cloud, 307,200 points (MeshLab, CloudCompare)
  table_map_top_down.png   top-down map with 10 cm grid
  overlay_check.png        model drawn back onto the photo
  depth_map.png            depth map (red near, blue far)
  novel_view_chair_side.png  point cloud from a new viewpoint
  measurements.csv         measurement table
  scene.json               camera, scale and every object's position, size and rotation
code/
  reconstruct.py           calibration, measurement, 3D model, point cloud
  make_map.py              top-down map and new-view render
  build_viewer.py          builds index.html from viewer_template.html
```

## Run it

```
pip install -r code/requirements.txt
python code/reconstruct.py
python code/make_map.py
python code/build_viewer.py
```

## Limitations

- Line detection, calibration, scale and the point cloud are automatic. The object corner, contact and top points for this photo are marked by hand in `reconstruct.py`, so a new photo needs new points.
- Only what the camera sees can be measured. Hidden parts (back of the bag, drawer unit depth, the bottle's base behind the laptop lid) are estimated.
- The photo was WhatsApp-compressed, so the camera settings were recovered from geometry instead of EXIF data.
