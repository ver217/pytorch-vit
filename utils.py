class Config(dict):
    def __init__(self, *args, **kwargs):
        assert len(args) <= 1, 'Config can only take a dict as argument'
        if len(args) > 0:
            dict_ = args[0]
            for k, v in dict_.items():
                if isinstance(v, dict):
                    v = Config(v)
                self[k] = v
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = Config(v)
            self[k] = v

    def __getattr__(self, name: str):
        return self[name]

    def __setattr__(self, name: str, value):
        self[name] = value
