import numpy as np




class VDNSampling:

    def __init__(self, satellites, batch_size=32):
        self.batch_size = batch_size
        self.satellites = satellites
        self.min_buff_size = np.min([len(sat['experience_buffer']) for sat in self.satellites])
        self.num_reels = min([len(sat['experience_reels']) for sat in self.satellites])

    def sample(self):
        if self.num_reels == 0:
            return self.sample_buffer()
        else:
            return self.sample_reels()

    def sample_buffer(self):
        sat_experiences = []
        rand_indices = np.random.randint(0, self.min_buff_size, size=self.batch_size).tolist()
        for sat in self.satellites:
            sat_experiences.append([sat['experience_buffer'][i] for i in rand_indices])
        return sat_experiences

    def sample_reels(self):
        sat_experiences = []
        min_reel_size = min([min([len(expr) for expr in sat['experience_reels']]) for sat in self.satellites])

        rand_reels = np.random.randint(0, self.num_reels, size=self.batch_size).tolist()
        rand_indices = np.random.randint(0, min_reel_size, size=self.batch_size).tolist()

        for sat in self.satellites:
            sat_experiences.append([sat['experience_reels'][r][i] for r, i in zip(rand_reels, rand_indices)])
        return sat_experiences







