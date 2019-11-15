import config

class TrivialAgent():

    def action(self, observation, t, threshold):

        paddleMid = int(observation[72]) + 13
        ballMid = int(observation[99]) + 1
        ballY = int(observation[101])

        if t == 0 or ballY > 200 or ballY == 0:
            action = config.ACTION_FIRE
        elif paddleMid - ballMid > threshold:
            action = config.ACTION_LEFT
        elif paddleMid - ballMid < - threshold:
            action = config.ACTION_RIGHT
        else:
            action = config.ACTION_REST

        return action
