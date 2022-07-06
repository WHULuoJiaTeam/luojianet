config = dict(
    model=dict(
        params=dict(
            in_channels=270,
            num_classes=9,
            pool_outsize1=(552, 400),
            pool_outsize2=(276, 200),
            pool_outsize3=(138, 100),
            pool_outsize4=(69, 50)
        )
    ),

    dataset=dict(
        type='S3ANet',
        params=dict(
            train_gt_dir="Matlab_data_format/WHU-Hi-LongKou/Training samples and test samples/Train50.mat",
            train_gt_name='LKtrain50',
            train_data_dir="Matlab_data_format/WHU-Hi-LongKou/WHU_Hi_LongKou.mat",
            train_data_name='WHU_Hi_LongKou',
            encoder_size=8
        )
    ),

    test=dict(
            type='S3ANet',
            params=dict(
                test_gt_dir="Matlab_data_format/WHU-Hi-LongKou/Training samples and test samples/Test50.mat",
                test_gt_name='LKtest50',
                test_data_dir="Matlab_data_format/WHU-Hi-LongKou/WHU_Hi_LongKou.mat",
                test_data_name='WHU_Hi_LongKou',
                encoder_size=8
            )
        ),

    save_model_dir='./saved_ckpts/S3ANet_LK.ckpt',
    num_class=9,
    image_shape=(550, 400),
    picture_save_dir='./saved_ckpts/S3ANet_LK.jpg',

)