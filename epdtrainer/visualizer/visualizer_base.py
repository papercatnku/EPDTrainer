

class visualizer_base:
    def __init__(self, if_train=False):
        self.if_train = if_train
        return

    def train(self):
        self.if_train = True

    def eval(self):
        self.if_train = False

    def get_eval_viz(self, data_dict):
        return {}

    def get_train_viz(self, data_dict):
        return {}

    def __call__(self, data_dict):
        viz_dict = {}

        if self.if_train:
            train_viz_dict = self.get_train_viz(data_dict)
            viz_dict.update(train_viz_dict)
        else:
            eval_viz_dict = self.get_eval_viz(data_dict)
            viz_dict.update(eval_viz_dict)

        return viz_dict
