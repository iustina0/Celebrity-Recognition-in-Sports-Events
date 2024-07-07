
class Config:
    data_dir = './FaceDetection/data/data/'
    min_size = 300
    max_size = 600
    num_workers = 4
    test_num_workers = 4

    rpn_sigma = 3.0
    roi_sigma = 1.0

    weight_decay = 0.0005
    lr_decay = 0.1
    lr = 1e-2

    plot_every = 100

    epoch = 20
    train_valid_num = 1000
    valid_split = 0.15

    test_num = 150
    load_path = None
    state_dict = None


opt = Config()
