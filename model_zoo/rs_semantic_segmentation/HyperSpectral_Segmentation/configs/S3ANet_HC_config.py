config = dict(
    model=dict(
        params=dict(
            in_channels=274,
            num_classes=16,
            pool_outsize1=(1224, 304),
            pool_outsize2=(612, 152),
            pool_outsize3=(306, 76),
            pool_outsize4=(153, 38)
        )
    ),

    dataset=dict(
        type='S3ANet',
        params=dict(
            train_gt_dir="Matlab_data_format/WHU-Hi-HanChuan/Training samples and test samples/Train50.mat",
            train_gt_name='Train50',
            train_data_dir="Matlab_data_format/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat",
            train_data_name='WHU_Hi_HanChuan',
            encoder_size=8
        )
    ),

    test=dict(
            type='S3ANet',
            params=dict(
                test_gt_dir="Matlab_data_format/WHU-Hi-HanChuan/Training samples and test samples/Test50.mat",
                test_gt_name='Test50',
                test_data_dir="Matlab_data_format/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat",
                test_data_name='WHU_Hi_HanChuan',
                encoder_size=8
            )
        ),

    save_model_dir='./saved_ckpts/S3ANet_HC.ckpt',
    num_class=16,
    image_shape=(1217, 303),
    picture_save_dir='./saved_ckpts/S3ANet_HC.jpg',
)