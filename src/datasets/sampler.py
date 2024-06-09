import numpy as np
import torch


# https://gist.github.com/usamec/1b3b4dcbafad2d58faa71a9633eea6a5
class ResumableSeedableSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, shuffle=True, seed=None):
        self.dataset = dataset
        self.generator = torch.Generator()
        self.seed = seed if seed is not None else np.random.randint(2**31)
        self.generator.manual_seed(self.seed)
        self.epoch = 0
        self.perm_index = 0
        self.shuffle = shuffle
        self.perm = self._get_perm()

    @property
    def num_samples(self):
        return len(self.dataset)

    def _get_perm(self):
        if self.shuffle:
            self.generator.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(self.num_samples, generator=self.generator)
        else:
            perm = torch.arange(self.num_samples)

        return perm

    def __iter__(self):
        while True:
            while self.perm_index < len(self.perm):
                self.perm_index += 1
                yield self.perm[self.perm_index - 1]

            if self.perm_index >= len(self.perm):
                self.perm_index = 0
                self.set_epoch(self.epoch + 1)
                self.perm = self._get_perm()

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def get_state(self):
        return {
            "perm": self.perm,
            "perm_index": self.perm_index,
            "seed": self.seed,
        }

    def set_state(self, state):
        self.perm = state["perm"]
        self.perm_index = state["perm_index"]
        self.seed = state["seed"]
