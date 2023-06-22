_base_ = ['./yolox_x_8xb4-12e_shift_from_clear_daytime.py']

# fp16 settings
model = dict(
    adapter = dict(
        optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
    )
)
test_cfg = dict(type='TestLoop', fp16=True)