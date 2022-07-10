import numpy as np


class Transition():
    def __init__(
        self,
        s_t: np.ndarray,
        a_t: int,
        r_t: float,
        s_next: np.ndarray,
        terminal: int
    ) -> None:
        super().__init__()
        self.s_t, self.a_t, self.r_t, self.s_next, self.terminal = s_t, a_t, r_t, s_next, terminal
        return


class basic():
    def __init__(
        self,
        capacity: int
    ) -> None:
        super().__init__()
        self.population = np.zeros(capacity)
        self.vaindex = 0
        self.capacity = capacity
        return

    @property
    def value(
        self
    ) -> np.ndarray:
        return self.population[:self.vaindex]

    def __len__(
        self
    ) -> int:
        return len(self.population)

    def append(
        self,
        target: object
    ) -> None:
        self.population[self.vaindex] = target
        if self.vaindex < self.capacity - 1:
            self.vaindex += 1
        return

    def evaluation_function(
        self,
        value: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError()

    def drop(
        self
    ) -> None:
        self.population[:-1] = self.population[1:]
        self.population[-1] = 0
        return

    def evaluate(
        self
    ) -> np.ndarray:
        return self.evaluation_function(self.value)

    def pop(
        self,
        popindex: int
    ) -> None:
        self.population[popindex:-1] = self.population[popindex + 1:]
        self.population[-1] = 0
        return


class Life(basic):
    def __init__(
        self,
        capacity: int
    ) -> None:
        super().__init__(capacity)
        return

    def evaluation_function(
        self,
        value: np.ndarray
    ) -> np.ndarray:
        v = np.max(value)
        if v == 0:
            v = 1
        return - value / v

    def modify_one(
        self
    ) -> None:
        self.population += 1
        return


class Totalreward(basic):
    def __init__(
        self,
        capacity: int
    ) -> None:
        super().__init__(capacity)
        self.reward_record = [0]
        return

    def evaluation_function(
        self,
        value: np.ndarray
    ) -> np.ndarray:
        rew = self.reward[np.uint8(value)]
        return rew / np.max(rew)

    @property
    def reward(
        self
    ) -> np.ndarray:
        return np.array(self.reward_record, dtype=float)


class Isterminate(basic):
    def __init__(
        self,
        capacity: int
    ) -> None:
        super().__init__(capacity)
        return

    def evaluation_function(
        self,
        value: np.ndarray
    ) -> np.ndarray:
        return 1 * value


class Timecount(basic):
    def __init__(
        self,
        capacity: int
    ) -> None:
        super().__init__(capacity)
        return

    def evaluation_function(
        self,
        value: np.ndarray
    ) -> np.ndarray:
        return np.exp(- value / np.average(value))


def main():
    return


if __name__ == "__main__":
    main()
