import numpy as np


class evaluator_base:
    def __init__(self, nm=None):
        if nm:
            self.nm = nm
            if self.nm[-1] not in '_-:':
                self.nm += '_'
        else:
            self.nm = ''
        return

    def feed_data(self, decode_res, data_dict):
        return

    def get_stastics_output(self,):
        dict_data_tolog = {}
        return dict_data_tolog

    def reset(self,):
        return

    def write_tblog(self, sw, tag='eval', step=None):
        return {}
