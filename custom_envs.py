import torch 

class RandomWalker1D:
    def __init__(self, loc=0.1, sd=0.02):
        self.state = torch.tensor([[0.0], [1.0]], dtype=torch.float)
        self.loc = loc
        self.sd = sd


        pass
    def step(self, a:int):
        noise = torch.normal(self.loc, self.sd, size=(1,), dtype=torch.float)
        if a == 0:
            next_pos = self.state[0, 0] + noise
        elif a == 1:
            next_pos = self.state[0, 0] - noise
        else:
            raise ValueError('a needs to be either integer 1 or 0.')
        terminated = False
        R = -1
        if next_pos > 1: # terminated and no position update
            terminated = True
            R = 1
        elif next_pos < -1: # Punish and no position update
            R = -2
        else:
            self.state[0, 0] = next_pos
            

        return self.state.clone(), R, terminated
    def reset(self):
        self.state = torch.tensor([[0.0], [1.0]], dtype=torch.float)
        return self.state.clone()
