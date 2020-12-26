from abc import ABC, abstractmethod


class RecurrentTrainer(ABC):
    @abstractmethod
    def get_update_info(self, update_policy_id, obs_batch, act_batch, avail_act_batch):
        raise NotImplementedError

    @abstractmethod
    def shared_train_policy_on_batch(self, update_policy_id, batch):
        raise NotImplementedError

    @abstractmethod
    def cent_train_policy_on_batch(self, update_policy_id, batch):
        raise NotImplementedError

    @abstractmethod
    def prep_training(self):
        raise NotImplementedError

    @abstractmethod
    def prep_rollout(self):
        raise NotImplementedError