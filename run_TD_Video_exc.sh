export MMPOSE=/media/mohammadreza/dados/crv_data/Padel/Padel_PoseDet_with_bash/mmpose
export DETCONFIGS=$MMPOSE/mmdetection/configs/
export DETCHK=$MMPOSE/mmdetection/checkpoints/
export POSECONFIGS=$MMPOSE/configs/body/2d_kpt_sview_rgb_img/
export POSECHK=$MMPOSE/checkpoints/
export TRCKCONFIGS=$MMPOSE/mmtracking/configs/mot/bytetrack/
export TRCKCHK=$MMPOSE/mmtracking/checkpoints/
export FILES=/media/mohammadreza/dados/crv_data/Padel/DATA_FOLDER
export SHEET=/media/mohammadreza/dados/crv_data/Padel/DATA_FOLDER
export COURT=/media/mohammadreza/dados/crv_data/Padel/Padel_PoseDet_with_bash/court_coord/Finals-Estrella_Damm_Open_2020-World_Padel_Tour.txt
export MODEL=/media/mohammadreza/dados/crv_data/Padel/Padel_PoseDet_with_bash/Padel_video_exc_version/Models
export COCO=$MMPOSE/tests/data/coco/test_coco.json
export CROWD=$MMPOSE/tests/data/crowdpose/test_crowdpose.json

## video processing
#mask_rcnn_r50_fpn_1x_coco
#python Video_TD_Padel.py $DETCONFIGS/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py $DETCHK/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth $POSECONFIGS/topdown_heatmap/coco/hrnet_w48_coco_256x192.py $POSECHK/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth $SHEET $FILES/Finals-Estrella_Damm_Open_2020-World_Padel_Tour.mp4 $COURT

# htc_r101_fpn_20e_coco
python Video_TD_Padel_exc.py $DETCONFIGS/htc/htc_r101_fpn_20e_coco.py $DETCHK/htc_r101_fpn_20e_coco_20200317-9b41b48f.pth $POSECONFIGS/topdown_heatmap/coco/hrnet_w48_coco_256x192.py $POSECHK/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth $TRCKCONFIGS/bytetrack_yolox_x_crowdhuman_mot17-private-half.py $TRCKCHK/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth $SHEET/Full-labeling1.xlsx $FILES/Finals-Estrella_Damm_Open_2020-World_Padel_Tour.mp4 $COURT $MODEL

# rtmdet
#python Video_TD_Padel.py $DETCONFIGS/rtmdet/rtmdet_r50_fpn_1x_coco.py $DETCHK/rtmdet_r50_fpn_1x_coco-329b1453.pth $POSECONFIGS/topdown_heatmap/coco/hrnet_w48_coco_256x192.py $POSECHK/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth $SHEET $FILES/Finals-Estrella_Damm_Open_2020-World_Padel_Tour.mp4 $COURT