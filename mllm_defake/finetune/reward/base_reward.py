from abc import ABC, abstractmethod


class BaseReward(ABC):
    """The base class for reward composition."""

    @abstractmethod
    def __init__(self):
        """Initialize the reward functions.

        Important: there must be two attributes: `num_functions` and `reward_names`.
        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self, pred, gt) -> dict:
        """Compute the reward.

        Args:
            pred: The prediction.
            gt: The ground truth.

        Returns:
            A dictionary of rewards, where the key is the reward name and the value is the reward value.
        """
        raise NotImplementedError
