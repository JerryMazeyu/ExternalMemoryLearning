class Opt():
    def __init__(self):
        self.batch_size = 16
        self.num_workers = 2
        self.shuffle = True
        self.device = 'cpu'

opt = Opt()