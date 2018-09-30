import json


class Parameters:
    def __init__(self,
                 diameter: int,
                 minmass: int,
                 maxmass: int,
                 maxsize: float,
                 search_range: int,
                 memory: int,
                 adjust_gamma: bool,
                 gamma: float,
                 use_clahe: bool,
                 clahe_clip_limit: float,
                 clahe_grid_size: tuple,
                 # use_arena_mask: bool,
                 circle_param1: float,
                 circle_param2: int,
                 circle_minradius: int,
                 circle_maxradius: int,
                 use_neural_network: bool,
                 neural_network_model_filepath: str,
                 params_name: str):

        self.parameters = locals().keys()

        self.diameter = diameter
        self.minmass = minmass
        self.maxmass = maxmass
        self.maxsize = maxsize
        self.search_range = search_range
        self.memory = memory
        self.adjust_gamma = adjust_gamma
        self.gamma = gamma
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        # self.use_arena_mask = use_arena_mask
        self.circle_param1 = circle_param1
        self.circle_param2 = circle_param2
        self.circle_minradius = circle_minradius
        self.circle_maxradius = circle_maxradius
        self.use_neural_network = use_neural_network
        self.neural_network_model_filepath = neural_network_model_filepath
        self.params_name = params_name

    def get_dict(self):
        d = {}
        for param in self.parameters:
            d.update({param: getattr(self, param)})
        return d

    @classmethod
    def from_json(cls, path):
        d = json.load(open(path))
        return cls(**d)
