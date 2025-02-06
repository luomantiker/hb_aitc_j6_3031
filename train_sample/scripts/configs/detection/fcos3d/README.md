# FCOS3D
|   model               |  dataset |   backbone     |   Input shape      |   config  |  ckpt download        |  demo download       |
| :----------:          | :-------:|  :--------:    |  :------------:    | :------:  |  :--------:           | :--------:           |
|   FCOS3D              | nuscenes | efficientnetb0 |   (512, 896)     | configs/detection/fcos3d/fcos3d_efficientnetb0_nuscenes.py | wget -c ftp://openexplorer@vrftp.horizon.ai/horizon_torch_samples/3.0.32/py310/modelzoo/qat_origin_modelzoo/fcos3d_efficientnetb0_nuscenes/* --ftp-password='c5R,2!pG' | wget -c ftp://openexplorer@vrftp.horizon.ai/horizon_torch_samples/3.0.32/py310/demo/fcos3d_efficientnetb0_nuscenes/* --ftp-password='c5R,2!pG' |

# Pretrained ckpt
|   backbone         |  ckpt download |
| :----------:       |  :-----------: |
| efficientnetb0     |  wget -c ftp://openexplorer@vrftp.horizon.ai/horizon_torch_samples/3.0.32/py310/modelzoo/qat_origin_modelzoo/fcos3d_efficientnetb0_nuscenes_pretrain/* --ftp-password='c5R,2!pG' |