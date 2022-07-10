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
        self.memory = SumTree(self.capacity)
        self.batch_size = hp.batch_size
        self.device = hp.device

        self.eplison = 0.01
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.0001

    def append(
        self,
        transition: Transition,
        **kwargs
    ) -> None:
        max_p = np.max(self.memory.tree[-self.capacity:])
        if max_p == 0:
            max_p = 1
        self.memory.add(max_p, transition)
        return

    def sample(
        self
    ) -> Transition:
        tree_idx_list, ISWeight, exp_sample = np.zeros(self.batch_size, dtype=np.int), np.zeros(self.batch_size), []
        # update beta
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        # more diversity
        priority_unit = self.memory.get_total_priority() / self.batch_size
        for i in range(self.batch_size):
            v = np.random.uniform(priority_unit * i, priority_unit * (i + 1))
            tree_idx, priority, transition = self.memory.get_leaf(v)
            prob = priority / self.memory.get_total_priority()
            ISW = np.power(self.capacity * prob, -self.beta)

            exp_sample.append(transition)
            tree_idx_list[i] = tree_idx
            ISWeight[i] = ISW
        ISWeight /= np.max(ISWeight)
        ISWeight = torch.from_numpy(ISWeight).float().to(self.device)

        S_t = torch.from_numpy(np.array([transition.s_t for transition in exp_sample])).float().to(self.device)
        A_t = torch.from_numpy(np.array([transition.a_t for transition in exp_sample])).long().to(self.device)
        R_t = torch.from_numpy(np.array([transition.r_t for transition in exp_sample])).float().to(self.device)
        S_next = torch.from_numpy(np.array([transition.s_next for transition in exp_sample])).float().to(self.device)
        Terminal = np.array([transition.terminal for transition in exp_sample])
        return Transition(S_t, A_t, R_t, S_next, Terminal), ISWeight, tree_idx_list

    def batch_update_priority(
        self,
        tree_idx_list: np.ndarray,
        TD_error_tensor: torch.Tensor
    ) -> None:
        td_error_list = TD_error_tensor.detach().cpu().numpy().flatten()
        td_error_list = np.abs(td_error_list) + self.eplison        # 1. convert to abs, avoid 0
        td_error_list = np.minimum(td_error_list, 1)                # 2. clipping
        priority_list = np.power(td_error_list, self.alpha)         # 3. td -> priority
        for tree_idx, priority in zip(tree_idx_list, priority_list):
            self.memory.update(tree_idx, priority)
        return

    def isFull(
        self
    ) -> bool:
        return False if 0 in self.memory.transition else True

    def getCount(
        self
    ) -> int:
        return np.count_nonzero(self.memory.transition)

    def clear(
        self
    ) -> None:
        self.memory = SumTree(self.capacity)
        return


class SumTree():
    def __init__(
        self,
        capacity: int
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.transition = np.zeros(capacity, dtype=object)
        self.transition_ptr = 0
        self.isFull = False
        return

    def add(
        self,
        priority: float,
        transition: Transition
    ) -> None:
        tree_idx = self.transition_ptr + self.capacity - 1
        self.transition[self.transition_ptr] = transition
        self.update(tree_idx, priority)
        self.transition_ptr += 1
        if self.transition_ptr >= self.capacity:
            self.transition_ptr = 0
            self.isFull = True
        return

    def update(
        self,
        tree_idx: int,
        priority: float
    ) -> None:
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += delta
        return

    def get_leaf(
        self,
        v: float
    ) -> (int, float, Transition):
        parent_idx = 0
        while True:
            L_child_idx = 2 * parent_idx + 1
            R_child_idx = L_child_idx + 1
            if L_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[L_child_idx]:
                    parent_idx = L_child_idx
                else:
                    v -= self.tree[L_child_idx]
                    parent_idx = R_child_idx
        transition_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.transition[transition_idx]

    def get_total_priority(
        self
    ) -> float:
        return self.tree[0]


def main():
    return


if __name__ == "__main__":
    main()
