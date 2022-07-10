import torch
import random
import numpy as np
from ExperienceReplay.Transition import Transition, Life, Totalreward, Isterminate, Timecount


class ReplayBuffer():
    def __init__(
        self,
        hp: object,
        sample_function: str = 'tournament'
    ) -> None:
        super().__init__()
        if sample_function == 'tournament':
            self.sample_function = self.tournament
        else:
            self.sample_function = self.roulette_wheel
        self.capacity = hp.capacity
        self.batch_size = hp.batch_size
        self.device = hp.device
        self.memory = []
        self.life = Life(hp.capacity)
        self.totalreward = Totalreward(hp.capacity)
        self.isterminate = Isterminate(hp.capacity)
        self.timecount = Timecount(hp.capacity)
        self.arg_dict = {'life': self.life,
                         'index': self.totalreward,
                         'done': self.isterminate,
                         'time': self.timecount}
        try:
            self.Max_R = hp.Max_R
        except AttributeError():
            self.Max_R = 9999999999
        return

    def isFull(
        self
    ) -> bool:
        return True if len(self.memory) >= self.capacity else False

    def append(
        self,
        transition: Transition,
        **kwargs
    ) -> None:
        if self.isFull():
            self.memory.pop(self.popindex)
            for i in self.arg_dict.values():
                i.pop(self.popindex)
        for k, v in kwargs.items():
            self.arg_dict.get(k, []).append(v)
        if kwargs['done'] or kwargs['time'] == self.Max_R:
            self.totalreward.reward_record[-1] = kwargs['r']
            self.totalreward.reward_record.append(0)
            self.life.modify_one()
        self.memory.append(transition)
        return

    def sample(
        self
    ) -> Transition:
        idx_sample = self.sample_function(self.batch_size)
        exp_sample = [self.memory[idx] for idx in idx_sample]
        S_t = torch.from_numpy(np.array([transition.s_t for transition in exp_sample])).float().to(self.device)
        A_t = torch.from_numpy(np.array([transition.a_t for transition in exp_sample])).long().to(self.device)
        R_t = torch.from_numpy(np.array([transition.r_t for transition in exp_sample])).float().to(self.device)
        S_next = torch.from_numpy(np.array([transition.s_next for transition in exp_sample])).float().to(self.device)
        Terminal = np.array([transition.terminal for transition in exp_sample])
        return Transition(S_t, A_t, R_t, S_next, Terminal)

    def getCount(
        self
    ) -> int:
        return len(self.memory)

    def clear(
        self
    ) -> None:
        self.memory = []
        return

    def roulette_wheel(
        self,
        batchsize
    ) -> list:
        index = random.choices(range(len(self.value)), self.value, k=batchsize)
        return index

    def tournament(
        self,
        batchsize: int,
        rand: bool = False
    ) -> list:
        index = []
        err = False
        if rand:
            err = random.random() < rand
        for i in range(batchsize):
            x = random.choice(range(len(self.value)))
            y = random.choice(range(len(self.value)))
            if self.value[x] > self.value[y] and not err:
                index.append(x)
            else:
                index.append(y)
        return index

    @property
    def value(
        self
    ) -> float:
        x = 0
        for i in self.arg_dict.values():
            x += i.evaluate()
        return x

    @property
    def popindex(
        self
    ) -> int:
        return np.argmin(self.value)


def main():
    return


if __name__ == "__main__":
    main()
