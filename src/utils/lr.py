'''A wrapper class for scheduled optimizer '''

class CustomExpLr():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, initial_learning_rate, decay_steps, decay_rate):
        self._optimizer = optimizer
        self.n_steps = 0
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        print('Using Custom exp lr-schedule')

    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.initial_learning_rate * self.decay_rate ** (self.n_steps / self.decay_steps)

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
    
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != '_optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)