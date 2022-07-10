import torch
import numpy as np
from ExperienceReplay.Transition import Transition


class ReplayBuffer():
    def __init__(
        self,
        hp: object
    ) -> None:
        super().__init__()
        self.capacity = hp.capacity
        self.batch_size = hp.batch_size
        self.device = hp.device
        self.memory = []
        return

    def isFull(
        self
    ) -> bool:
        return True if self.getCount() >= self.capacity else False

    def append(
        self,
        transition: Transition,
        **kwargs
    ) -> None:
        if self.isFull():
            self.memory.pop(0)
        self.memory.append(transition)
        return

    def sample(
        self
    ) -> Transition:
        idx_sample = np.random.randint(low=0, high=self.getCount(), size=self.batch_size)
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


def main():
    return


if __name__ == "__main__":
    main()
