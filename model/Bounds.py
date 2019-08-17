

class Bounds(object):

    def __init__(self, reset_count = 1000000):

        self.min = 99999.9
        self.max = 0.0
        self.future_min = 99999.9
        self.future_max = 0.0
        self.update_count = 0
        self.reset_count = reset_count


    def update_bounds(self, value, continuous = True):
        if not value <= 0:
            self.future_min = min(self.future_min, value)
            if continuous:
                self.min = min(self.min, value)
        self.future_max = max(self.future_max, value)
        if continuous:
            self.max = max(self.max, value)
        self.update_count+=1
        # if (self.update_count+1) % self.reset_count == 0:
        #     self.update_count = 0
        #     self.reset_future_bounds()


    def get_bounds(self, future=False):
        if future:
            return [self.future_min,self.future_max]
        return [self.min, self.max]

    def reset_future_bounds(self):
        self.min = self.future_min
        self.max = self.future_max
        self.future_max = 0.0
        self.future_min = 99999.9