import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ExperienceReplay import ER, PER, EER, Transition
from DQN_farm.HyperParameter import HyperParameter_CartPole as HP


class Q_Net(nn.Module):
    def __init__(
        self,
        hp: HP
    ) -> None:
        super(Q_Net, self).__init__()
        hidden_size = 50
        self.fc1 = nn.Linear(hp.state_space, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_size, hp.action_space)
        self.out.weight.data.normal_(0, 0.1)
        return

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        q_value = self.out(x)
        return q_value


class DQN():
    def __init__(
        self,
        state_space: int,
        action_space: int,
        ER_type: str,
        sample_function: str = 'tournament'
    ) -> None:
        super().__init__()
        self.hp = HP(state_space, action_space)
        self.Q_target = Q_Net(self.hp).to(self.hp.device)
        self.Q_behavior = Q_Net(self.hp).to(self.hp.device)
        self.ER_type = ER_type
        if self.ER_type == 'ER':
            self.ER = ER.ReplayBuffer(self.hp)
        elif self.ER_type == 'PER':
            self.ER = PER.ReplayBuffer(self.hp)
        elif self.ER_type == 'EER':
            self.ER = EER.ReplayBuffer(self.hp, sample_function)
        else:
            raise NotImplementedError()

        self.optimizer = torch.optim.Adam(self.Q_behavior.parameters(), lr=self.hp.lr)
        self.learning_iter = 0
        return

    def store_experience(
        self,
        transition: Transition,
        **kwargs
    ) -> None:
        self.ER.append(transition, **kwargs)
        return

    def get_action(
        self,
        s_t: torch.Tensor,
        EPSILON: float
    ) -> None:
        if np.random.uniform() < EPSILON:
            action = np.random.randint(low=0, high=self.hp.action_space)
        else:
            s_t = torch.unsqueeze(torch.FloatTensor(s_t), 0).to(self.hp.device)
            q_value = self.Q_behavior(s_t)
            action = torch.argmax(q_value).cpu().numpy()
        return action

    def get_action_target(
        self,
        s_t: torch.Tensor
    ) -> torch.Tensor:
        s_t = torch.unsqueeze(torch.FloatTensor(s_t), 0).to(self.hp.device)
        q_value = self.Q_target(s_t).detach()
        action = torch.argmax(q_value).cpu().numpy()
        return action

    def learn(
        self
    ) -> torch.Tensor:
        # update Q_target (copy)
        if self.learning_iter % self.hp.target_replace_iter == 0:
            self.Q_target.load_state_dict(self.Q_behavior.state_dict())
        self.learning_iter += 1

        # sample from ER
        if self.ER_type == 'ER' or self.ER_type == 'EER':
            T = self.ER.sample()
        elif self.ER_type == 'PER':
            T, ISWeight, tree_idx_list = self.ER.sample()
        S_t, A_t, R_t, S_next, Terminal = T.s_t, T.a_t, T.r_t, T.s_next, T.terminal

        # update Q_behavior (TD)
        q_t = self.Q_behavior(S_t).gather(1, A_t.view(-1, 1))
        q_next = self.Q_target(S_next).detach()
        GAMMA = torch.from_numpy(self.hp.GAMMA * (1 - Terminal)).float().to(self.hp.device)
        q_target = (R_t + GAMMA * q_next.max(1)[0]).view(-1, 1)

        loss = (q_target - q_t).pow(2)
        if self.ER_type == 'PER':
            self.ER.batch_update_priority(tree_idx_list, loss)
            loss = (q_target - q_t).pow(2) * ISWeight
        loss = loss.mean()

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(
        self,
        filename: str
    ) -> None:
        torch.save(self.Q_target, filename+'_target.pkl')
        torch.save(self.Q_behavior, filename+'_behavior.pkl')
        return


def main():
    return


if __name__ == "__main__":
    main()
