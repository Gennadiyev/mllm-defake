from abc import ABC, abstractmethod


class BaseReward(ABC):
    """The base class for reward composition."""

    def __init__(self):
        """Initialize the reward functions.

        super().__init__() should be called after the initialization of the subclass.
        Important: there must be two attributes: `num_functions` and `reward_names`.
        """
        if not hasattr(self, "num_functions"):
            raise AttributeError("The number of functions must be specified.")
        if not hasattr(self, "reward_names"):
            raise AttributeError("The reward names must be specified.")

    @abstractmethod
    def __call__(self, user_input: str, assistant_output: str, completion: str) -> dict:
        """Compute the reward.

        Args:
            user_input: The user input.
            assistant_output: The assistant output (the ground truth).
            completion: The completion.

        Returns:
            A dictionary of rewards.
            Keys and values:
                - "all_reward": (float), the final reward.
                - "per_function_reward": (list[float]), the reward for each function.
                    The order of the functions is the same as the order of `reward_names`.
        """
        raise NotImplementedError
