_base_ = ['./yolox_x_8xb4-24e_shift_daytime.py']

# fp16 settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
test_cfg = dict(type='TestLoop', fp16=True)