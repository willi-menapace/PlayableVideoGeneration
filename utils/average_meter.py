class AverageMeter:
    def __init__(self, *keys):
        self.data = dict()
        for k in keys:
            self.data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            if k not in self.data:
                self.data[k] = [0.0, 0]
            self.data[k][0] += v
            self.data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.data[keys[0]][0] / self.data[keys[0]][1]
        else:
            v_list = [self.data[k][0] / self.data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.data.keys():
                self.data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.data[key] = [0.0, 0]
            return v