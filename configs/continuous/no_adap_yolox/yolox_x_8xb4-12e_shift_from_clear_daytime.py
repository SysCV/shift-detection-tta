_base_ = [
    '../../_base_/models/yolox_x_8x8.py',
    '../../_base_/default_runtime.py'
]

dataset_type = 'SHIFTDataset'
data_root = 'data/shift/'
attributes = dict(weather_coarse='clear', timeofday_coarse='daytime')

img_scale = (800, 1440)
batch_size = 2

model = dict(
    type='AdaptiveDetector',
    data_preprocessor=dict(
        type='mmtrack.TrackDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='mmdet.BatchSyncRandomResize',
                random_size_range=(576, 1024),
                size_divisor=32,
                interval=10)
        ]),
    detector=dict(
        _scope_='mmdet',
        bbox_head=dict(num_classes=6),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'  # noqa: E501
        )),
    adapter=None)

train_pipeline = [
    dict(
        type='mmdet.Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        bbox_clip_border=False),
    dict(
        type='mmdet.RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        bbox_clip_border=False),
    dict(
        type='mmdet.MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        bbox_clip_border=False),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.Resize',
        scale=img_scale,
        keep_ratio=True,
        clip_object_border=False),
    dict(
        type='mmdet.Pad',
        size_divisor=32,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(type='mmtrack.PackTrackInputs', pack_single_img=True)
]
test_pipeline = [
    dict(type='LoadImageFromFile',
         backend_args=dict(
             backend='tar',
             tar_path=data_root + 'continuous/videos/1x/val/front/img_decompressed.tar',
         )
    ),
    dict(type='mmtrack.LoadTrackAnnotations'),
    dict(type='mmdet.Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='mmdet.Pad',
        size_divisor=32,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='mmtrack.PackTrackInputs', pack_single_img=True)
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='mmdet.MultiImageMixDataset',
    dataset=dict(
        type='SHIFTDataset',
        load_as_video=False,
        ann_file=data_root + 'discrete/images/train/front/det_2d_cocoformat.json',
        data_prefix=dict(img=''),
        ref_img_sampler=None,
        metainfo=dict(classes=('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')),
        pipeline=[
            dict(type='LoadImageFromFile', 
                backend_args=dict(
                    backend='zip',
                    zip_path=data_root + 'discrete/images/train/front/img.zip',
                )
            ),
            dict(type='mmtrack.LoadTrackAnnotations'),
        ],
        filter_cfg=dict(
            attributes=attributes,
            filter_empty_gt=False,
            min_size=32
        )),
    pipeline=train_pipeline)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

val_dataset=dict(
    type='SHIFTDataset',
    load_as_video=True,
    ann_file=data_root + 'continuous/videos/1x/val/front/det_2d_cocoformat.json',
    # ann_file=data_root + 'continuous/videos/1x/val/front/det_2d_cocoformat_tmp.json',
    data_prefix=dict(img=''),
    ref_img_sampler=None,
    test_mode=True,
    filter_cfg=dict(attributes=attributes),
    pipeline=test_pipeline,
    metainfo=dict(classes=('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='mmtrack.VideoSampler'),
    dataset=val_dataset)
test_dataloader = val_dataloader
# optimizer
# default 8 gpu
lr = 0.0005 / 8 * batch_size
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))

# some hyper parameters
# training settings
total_epochs = 12
num_last_epochs = 2
resume_from = None
interval = 5

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning policy
param_scheduler = [
    dict(
        # use quadratic formula to warm up 1 epochs
        # and lr is updated by iteration
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 1 to epoch #(total_epochs - num_last_epochs) 
        type='mmdet.CosineAnnealingLR',
        eta_min=lr * 0.05,
        begin=1,
        T_max=total_epochs - num_last_epochs,
        end=total_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 10 epochs
        type='mmdet.ConstantLR',
        by_epoch=True,
        factor=1,
        begin=total_epochs - num_last_epochs,
        end=total_epochs,
    )
]

custom_hooks = [
    dict(
        type='mmtrack.YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(type='mmdet.SyncNormHook', priority=48),
    dict(
        type='mmdet.EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]
default_hooks = dict(checkpoint=dict(interval=1))

# evaluator
val_evaluator = [
    dict(type='SHIFTVideoMetric', metric=['bbox'], classwise=True),
]
test_evaluator = val_evaluator