import random
from typing import List


class Bottle:

    def __init__(self, colors: List[str]) -> None:
        self.colors = colors

    def __str__(self):
        return "".join(self.colors)

    def __len__(self):
        return len(self.colors)

    def get_top(self):
        return self.colors[-1]

    def peek_next(self):
        pass

    def pop_next(self):
        pass


class StackBottle(Bottle):

    def __init__(self, colors) -> None:
        Bottle.__init__(self, colors)

    def peek_next(self) -> str:
        if len(self.colors) == 0:
            raise ValueError("No colors to peek.")
        return self.colors[-1]

    def pop_next(self) -> str:
        if len(self.colors) == 0:
            raise ValueError("No colors to pop.")

        return self.colors[-1]


class QueueBottle(Bottle):

    def __init__(self, colors) -> None:
        Bottle.__init__(self, colors)

    def peek_next(self) -> str:
        if len(self.colors) == 0:
            raise ValueError("No colors to peek.")
        return self.colors[0]

    def pop_next(self) -> str:
        if len(self.colors) == 0:
            raise ValueError("No colors to pop.")

        return self.colors[0]


class WaterLevelGenerator:

    def __init__(self, bottle_type: str, num_bottles: int) -> None:
        self.colors = [
            "Q",
            "W",
            "E",
            "R",
            "T",
            "Y",
            "U",
            "I",
            "O",
            "P",
            "A",
            "S",
            "D",
            "F",
            "G",
            "H",
        ]
        self.MAX_DEPTH = 4
        self.num_bottles = num_bottles

        if bottle_type == "stack":
            self.bottle_type = StackBottle
        elif bottle_type == "queue":
            self.bottle_type = QueueBottle
        else:
            raise ValueError("Bottle Type {bottle_type} not implemented.")

        self.init()

    def init(self) -> None:
        if self.num_bottles > len(self.colors):
            raise ValueError("Too many bottle, not enough colors!")

        out = [self.colors[i] for i in range(self.num_bottles) for _ in range(self.MAX_DEPTH)]
        random.shuffle(out)
        bottles = [
            self.bottle_type(self.colors[i : i + self.MAX_DEPTH]) for i in range(self.num_bottles)
        ]
        bottles.append(self.bottle_type([]))
        bottles.append(self.bottle_type([]))

        self.bottles = bottles

    def show_bottles(self):
        for bottle in self.bottles:
            out = str(bottle)
            out = out + "-" * (self.MAX_DEPTH - len(bottle))
            print(out)


if __name__ == "__main__":
    level = WaterLevelGenerator(bottle_type="stack", num_bottles=8)
    level.show_bottles()
