grad_max_norm = 35
print_freq = 10
max_epochs = 200
warmup_iters = 200
# return_len_ = 10
return_len_ = 11

multisteplr = False
multisteplr_config = dict(
    decay_t = [87 * 500],
    decay_rate = 0.1,
    warmup_t = warmup_iters,
    warmup_lr_init = 1e-6,
    t_in_epochs = False
)


optimizer = dict(
    optimizer=dict(
        type='AdamW',
        lr=1e-3,
        weight_decay=0.01,
    ),
)

data_path = 'data/nuscenes/'


train_dataset_config = dict(
    type='nuScenesSceneDatasetLidar',
    data_path = data_path,
    return_len = return_len_, 
    offset = 0,
    imageset = 'data/nuscenes_infos_train_temporal_v3_scene.pkl', 
)
    
val_dataset_config = dict(
    type='nuScenesSceneDatasetLidar',
    data_path = data_path,
    return_len = return_len_, 
    offset = 0,
    imageset = 'data/nuscenes_infos_val_temporal_v3_scene.pkl', 
)

train_wrapper_config = dict(
    type='tpvformer_dataset_nuscenes',
    phase='train', 
)

val_wrapper_config = dict(
    type='tpvformer_dataset_nuscenes',
    phase='val', 
)

train_loader = dict(
    batch_size = 1,
    shuffle = True,
    num_workers = 1,
)

val_loader = dict(
    batch_size = 1,
    shuffle = False,
    num_workers = 1,
)

    

loss = dict(
    type='MultiLoss',
    loss_cfgs=[
        dict(
            type='ReconLoss',
            weight=10.0,
            ignore_label=-100,
            use_weight=False,
            cls_weight=None,
            input_dict={
                'logits': 'logits',
                'labels': 'inputs'}),
        dict(
            type='LovaszLoss',
            weight=1.0,
            input_dict={
                'logits': 'logits',
                'labels': 'inputs'}),
        # dict(
        #     type='VQVAEEmbedLoss',
        #     weight=1.0),

        dict(
            type='KldLoss',
            weight=0.002,
            input_dict={
                'z_mu': 'z_mu',
                'logvar': 'logvar'}),
        ])

# loss_input_convertion = dict(
#     logits='logits',
#     embed_loss='embed_loss'
# )

loss_input_convertion = dict(
    logits='logits',
    # embed_loss='embed_loss'
    z_mu='z_mu',
    logvar='logvar'
)


load_from = ''

_dim_ = 16
expansion = 8
base_channel = 64
# base_channel = 96

n_e_ = 512
model = dict(
    # type = 'VAERes2D',
    type = 'CustomVAERes2D',
    encoder_cfg=dict(
        # type='Encoder2D',
        type='CustomEncoder2D',

        ch = base_channel, 
        out_ch = base_channel, 
        # ch_mult = (1,2,4), 
        ch_mult = (1,2,4,8), 

        # num_res_blocks = 2,
        num_res_blocks = 4,
        # attn_resolutions = (50,), 
        attn_resolutions = (25,), 
        dropout = 0.0, 
        resamp_with_conv = True, 
        in_channels = _dim_ * expansion,
        resolution = 200, 
        # z_channels = base_channel * 2, 
        # double_z = False,
        z_channels = 32, 
        double_z = True,

    ), 
    decoder_cfg=dict(
        # type='Decoder2D',
        type='CustomDecoder2D',

        ch = base_channel, 
        out_ch = _dim_ * expansion, 
        # ch_mult = (1,2,4), 
        ch_mult = (1,2,4,8), 
        num_res_blocks = 2,
        # attn_resolutions = (50,), 
        attn_resolutions = (25,), 
        dropout = 0.0, 
        resamp_with_conv = True, 
        in_channels = _dim_ * expansion,
        resolution = 200, 
        # z_channels = base_channel * 2, 
        # z_channels = base_channel, 
        z_channels = 32, 
        give_pre_end = False
    ),
    # decoder_cfg=dict(
    #     # type='Decoder2D',
    #     type='CustomTemporalDecoder',

    #     ch = base_channel, 
    #     out_ch = _dim_ * expansion, 
    #     # ch_mult = (1,2,4), 
    #     ch_mult = (1,2,4,8), 
    #     num_res_blocks = 2,
    #     # attn_resolutions = (50,), 
    #     attn_resolutions = (25,), 
    #     dropout = 0.0, 
    #     resamp_with_conv = True, 
    #     in_channels = _dim_ * expansion,
    #     resolution = 200, 
    #     # z_channels = base_channel * 2, 
    #     # z_channels = base_channel, 
    #     z_channels = 32, 
    #     give_pre_end = False
    # ),
    num_classes=18,
    expansion=expansion, 
    # vqvae_cfg=dict(
    #     type='VectorQuantizer',
    #     n_e = n_e_, 
    #     e_dim = base_channel * 2, 
    #     beta = 1., 
    #     z_channels = base_channel * 2, 
    #     use_voxel=False)
    )

shapes = [[200, 200], [100, 100], [50, 50], [25, 25]]

unique_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
label_mapping = "./config/label_mapping/nuscenes-occ.yaml"