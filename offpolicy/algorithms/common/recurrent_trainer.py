from abc import ABC, abstractmethod


class RecurrentTrainer(ABC):
    """Abstract trainer class. Performs gradient updates to policies.."""
    @abstractmethod
    def train_policy_on_batch(self, update_policy_id, batch):
        """
        Performs a gradient update for the specified policy using a batch of sampled data.
        :param update_policy_id: (str) id of policy to update.
        :param batch (Tuple) batch of data sampled from buffer, containing
        """
        raise NotImplementedError

    @abstractmethod
    def prep_training(self):
        raise NotImplementedError

    @abstractmethod
    def prep_rollout(self):
        raise NotImplementedError