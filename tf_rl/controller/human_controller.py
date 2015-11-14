from tf_rl.utils.getch import getch
from redis import StrictRedis



class HumanController(object):
    def __init__(self, mapping):
        self.mapping = mapping
        self.r = StrictRedis()
        self.experience = []

    def action(self, o):
        return self.mapping[self.r.get("action")]

    def store(self, observation, action, reward, newobservation):
        pass

    def training_step(self):
        pass



def control_me():
    r = StrictRedis()
    while True:
        c = getch()
        r.set("action", c)


if __name__ == '__main__':
    control_me()
