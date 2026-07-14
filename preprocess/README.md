# ITRI-campus 前處理 (preprocess)

把自拍的 `itri_campus` 資料轉成 GCMLoc 模型可用的格式，輸出到
`../iter_campus/`。設計上對齊 Argoverse 的資料流程
（點雲地圖 + 每幀 `cam_T_map` 位姿 + pinhole 影像），
所以之後寫 `Dataset_itri_*.py` 時可直接沿用 Argoverse loader 的轉換 / crop 邏輯。

> 注意：前相機影像 `camera/...f_hdr.h265/*.jpg` **本身已經是去畸變後的
> pinhole 影像**（邊緣是直的，底部有魚眼校正留下的灰色弧形填充），所以
> **不需要再做去畸變**。直接使用原圖，內參取 calib json 的 `projection_matrix`
> P (= `kitti_format/calib.txt` 的 P0)。影像直接從 `raw/sequences/<seq>/camera_front`
> 讀取，不額外複製。

## 產出結構

```
iter_campus/
  raw/                              ← 只連結「我們真的會用到」的來源資料 (symlink)
    pcd_map -> itri_campus/pcd_map
    submaps_config.json -> ...
    sequences/<seq>/
      camera_front     -> .../camera/lucid_cameras_x00.gige_100_f_hdr.h265
      calib_front.json -> .../calib/camera/...f_hdr.h265.json
      kitti_format     -> .../kitti_format
  processed/                        ← 前處理生成的檔案 (真實檔案)
    map.h5                          ← 全域點雲地圖 (4,M)+intensity (驗證/策略A用)
    sequences/<seq>/
      poses_torch/<ts>.npy          ← 4x4 cam_T_map (每幀相機外參)
      pinhole_calib.json            ← {fx,fy,cx,cy,width,height} (來自 P)
      frame_index.csv               ← idx,time_s,img_ts_ns,dt_ms
      overlay/<ts>.jpg              ← (驗證用) 點雲投影到原圖的疊圖
```

影像不複製，直接讀 `raw/sequences/<seq>/camera_front/<ts>.jpg`（已是 pinhole 原圖）。

`raw/` 全是 symlink，一眼就能看出這份研究依賴哪些來源資料、其餘
(其他 3 顆相機、panorama、h265 影片、tf、semantic map、combined_data.json) 都不需要。

> 地圖策略：本流程預設仍會建 `map.h5`（全域降採樣，方便驗證與策略 A）。
> 已選定 **策略 C**（loader 執行時依位姿查 `pcd_map` 的 50m 鄰近子地圖、保留
> 5cm 全解析度），該邏輯會寫在 `Dataset_itri_*.py`，不需額外前處理檔
> （子地圖中心已在 `submaps_config.json`）。

## 一鍵執行

```bash
cd preprocess
./run_all.sh                      # 全部序列
VOXEL=0.2 ./run_all.sh
```

或逐步執行（全部用 `conda run --no-capture-output -n CMRNet_4090 python`）：

| 步驟 | 腳本 | 說明 |
|---|---|---|
| 1 | `build_links.py` | 建立 `iter_campus/raw/` symlink 視圖 |
| 2 | `build_map.py --voxel 0.15` | 合併 205 個 `vg05_filtered` submap → `map.h5`，並降採樣 (原始 119M 點 @5cm 太大)。策略 C 下此步僅供驗證 |
| 3 | `build_calib.py` | 由 calib json 的 `projection_matrix` 寫每序列 `pinhole_calib.json`（影像已是 pinhole，不去畸變）|
| 4 | `build_poses.py` | 由 **100Hz tf 軌跡內插**到每張影像時刻算 `cam_T_map`（不用 `poses.txt`）|
| 4b | `build_split.py` | 讀框選多邊形 + `frame_index.csv` → 烤出 `splits.json`（train/test 時間戳清單）|
| 4c | `make_split_view.py` | 依 `splits.json` 建 symlink 的 train/test 影像分組供肉眼檢視 |
| 驗 | `verify_overlay.py --seq <名稱> --num 6` | 把 `map.h5` 投影到**原圖**疊圖，目視確認對齊 |

常用參數：
- `build_map.py --voxel`：輸出地圖體素大小 (m)，越大越省記憶體；0.15 約留下 23.5M 點。
- `verify_overlay.py --alpha`：點透明度 0..1（越低背景越清楚）；`--point-size`：點大小。
- `build_poses.py --pose-frame {lidar,camera}`：見下方座標說明。

## 座標與位姿（重點）

- **pose 來源 = 100Hz `tf/map.base_link` 軌跡，內插到每張影像時刻**。
  ⚠️ **不用 `kitti_format/poses.txt`**：它是自建檔，經比對在 itri_1（~0.9m）、
  typhoon（~1.5m）與官方軌跡不一致（因 poses.txt 與 times.txt 行數不符、配對位移）。
- `tf/map.base_link/*.json` 的 `extrinsic` E = **`base_link_T_map`**（map→base_link）；
  `inv(E)` = `map_T_base`（車輛在 map 的位姿，平移 z≈138，與 `pcd_map` 同一個
  局部地理座標系，原點 lat 24.775 / lon 121.046）。
- `calib.txt` 的 `Tr_velo_to_cam` 經驗證 **等於 `cam_T_base_link`**
  （與 `tf_static/to_base_link` 前相機外參互逆，誤差 0.0）。
- 每幀相機外參：

  ```
  map_T_base(t_img) = 內插( inv(E) , 影像時刻 t_img )   # 位置 lerp + 旋轉 SLERP
  cam_T_map        = Tr_velo_to_cam @ inv(map_T_base(t_img))
  ```

  地圖點 `X_map` 經 `X_cam = cam_T_map @ X_map` 得到光學系座標
  (x 右、y 下、z 前)，正好是 Argoverse loader 軸序重排前的慣例。
- 內插準確度（leave-one-out 驗證）：位置誤差 mean 0.2–2.5mm、max ~1.6cm；
  旋轉 mean ~0.01°、max ~0.12° → 遠低於定位本身噪聲。
- 投影內參用 `pinhole_calib.json`（= `projection_matrix` P：
  fx=540.27, fy=539.78, cx=766.53, cy=453.72，4 序列相同）。
- `frame_index.csv` 欄位：`idx, img_ts_ns, bracket_ms, map_x, map_y`
  （`bracket_ms`=內插用的前後 tf 間隔；`map_x/map_y`=車輛在 map 的位置，供切分用）。

## 驗證怎麼看

`verify_overlay.py` 會把地圖點投影到**原圖**、依深度上色 (近紅遠藍)。
若點貼合路面/建物/桿件 → 座標鏈正確；若整體明顯平移或旋轉 →
改用 `build_poses.py --pose-frame camera` 重跑，或檢查 `Tr_velo_to_cam`。

## 之後接到訓練

前處理完成後，下一階段會新增 `Dataset_itri_mapping.py` /
`Dataset_itri_localization.py`（仿 Argoverse 兩支），並在
`train_ablation.py` / `train_loc_ablation.py` 加入 `datasetType == 2` 分支。
本資料夾只負責把資料準備好。
