from abc import ABC, abstractmethod


class RecurrentTrainer(ABC):
    @abstractmethod
    def train_policy_on_batch(self, update_policy_id, batch):
        raise NotImplementedError

    @abstractmethod
    def prep_training(self):
        raise NotImplementedError

    @abstractmethod
    def prep_rollout(self):
        raise NotImplementedError