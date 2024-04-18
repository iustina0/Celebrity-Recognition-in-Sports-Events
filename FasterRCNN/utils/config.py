
class Config:
    data_dir = './data/data/'
    min_size = 300
    max_size = 600
    num_workers = 4
    test_num_workers = 4

    rpn_sigma = 3.
    roi_sigma = 1.

    weight_decay = 0.0005
    lr_decay = 0.1
    lr = 1e-3

    env = 'fasterrcnn'
    port = 8097
    plot_every = 100

    data = 'voc'
    pretrained_model = 'vgg16'

    epoch = 14
    train_valid_num = 1000
    valid_split = 0.15

    test_num = 150

    debug_file = '/tmp/debugf'

    load_path = None

    def _parse(self, kwargs):
        state_dict = self.state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        print(self.state_dict())
        print('==========end============')

    def state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() if not k.startswith('_')}


opt = Config()
