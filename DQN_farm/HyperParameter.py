import torch


class HyperParameter_CartPole():
    def __init__(
        self,
        state_space: int,
        action_space: int
    ) -> None:
        super().__init__()
        # select device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # model parameter
        self.state_space = state_space
        self.action_space = action_space
        # training parameter
        self.lr = 0.005
        self.batch_size = 32
        self.target_replace_iter = 100
        self.GAMMA = 0.99
        # experience replay parameter
        # self.capacity = 1
        self.capacity = 4096
        # self.capacity = 65536
        self.Max_R = 200
        return


def main():
    return


if __name__ == "__main__":
    main()
