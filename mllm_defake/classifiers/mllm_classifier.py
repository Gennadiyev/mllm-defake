import importlib
import random
import sys
import warnings
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm.auto import tqdm

from mllm_defake.utils import encode_image_to_base64
from mllm_defake.vllms import VLLM

warnings.filterwarnings("ignore")
logger.remove()
logger.add(sys.stderr, level="INFO")
# Add . to sys.path to allow importing modules from the current directory

random.seed(3804)


class MLLMClassifier:
    @logger.catch
    def __init__(
        self,
        prompt: dict,
        model: VLLM,
        real_samples: list[Path],
        fake_samples: list[Path],
    ):
        """
        Initialize the MLLMClassifier with the given prompt, model, and samples. Run `evaluate` method to evaluate the model.
        """
        if not isinstance(prompt, dict) or "format_version" not in prompt or prompt.get("format_version") != "3":
            raise ValueError(f"Invalid prompt. Expected a v3 JSON object, but got: {prompt}")
        if not isinstance(model, VLLM) or not hasattr(model, "infer_raw"):
            raise ValueError(f"Invalid model. Expected a VLLM object with `infer_raw` method, but got: {model}")
        if not all(isinstance(x, Path) for x in real_samples):
            raise ValueError(f"Invalid real_samples. Expected a list of Path objects, but got: {real_samples}")
        if not all(isinstance(x, Path) for x in fake_samples):
            raise ValueError(f"Invalid fake_samples. Expected a list of Path objects, but got: {fake_samples}")
        self.prompt = prompt
        self.model = model
        self.real_samples = real_samples
        self.fake_samples = fake_samples
        self.all_samples = real_samples + fake_samples
        self.all_labels = [1] * len(real_samples) + [0] * len(
            fake_samples
        )  # Follows CNNDetection convention of 1=real, 0=fake
        self.samples = list(zip(self.all_samples, self.all_labels, strict=False))
        random.shuffle(self.samples)

    def get_decorator_func(self, decorator: str) -> callable:
        """
        Decorators should be a dictionary of decorator names and their corresponding callable functions that modifies the cache dict during inference. The callable functions should have the signature `def decorator(cache: dict) -> None`.
        """
        module_name, func_name = decorator.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name, package="mllm_defake.decorators")
            return getattr(module, func_name)
        except ImportError:
            try:
                module = importlib.import_module(f"decorators.{module_name}")
                return getattr(module, func_name)
            except ImportError:
                try:
                    module = importlib.import_module(module_name)
                    return getattr(module, func_name)
                except ImportError as e:
                    logger.error(
                        "Decorator module not found: {}; sys.path = {}",
                        module_name,
                        sys.path,
                    )
                    raise ImportError(f"Decorator module not found: {module_name} ({e})") from e

    def decorate(self, cache: dict, decorator: str) -> None:
        """
        Decorates the cache dict with the decorator function.

        Raises ImportError if the decorator module is not found.
        """
        decorator_func = self.get_decorator_func(decorator)
        if decorator_func is None:
            raise ImportError(f"The function {decorator} could not be imported")
        decorator_func(cache)

    def llm_query(self, sample: Path, label: int | bool, should_print_response: bool = False) -> str:
        """
        Query the LLM model with the given sample and label.

        This method processes the prompt and returns the raw response from the model.

        The `sample` argument should be a Path object to the image file.

        The `label` argument should be an integer (-1[unk], 0[fake] or 1[real]) or a boolean (False or True) indicating whether the sample is fake or real.

        The `should_print_response` argument should be a boolean indicating whether the model response should be printed to the console. Intended for debugging purposes.
        """
        # For caching model responses within this classification job, keys should be `response_var_name` values
        cache = {}
        # Prefill cache with sample and label
        cache["_image_path"] = sample
        cache["image_url"] = encode_image_to_base64(sample)
        cache["label"] = (
            "real"
            if (label == 1 or label is True or sample in self.real_samples)
            else "fake"
            if (label == 0 or label is False or sample in self.fake_samples)
            else "unknown"
        )  # Converts 0/1 to fake/real, and True/False to real/fake

        def replace_string_with_cache(string: str) -> str:
            # Works like f-string, but uses cache values
            return string.format(**cache)

        r = replace_string_with_cache
        for i, conv in enumerate(self.prompt["conversations"]):
            logger.debug(f"Processing conversation #{i} ({conv['id']})")
            messages = []
            # Process decorators
            if "decorators" in conv:
                for decorator in conv["decorators"]:
                    self.decorate(cache, decorator)
            for payload_item in conv["payload"]:
                role, content = payload_item[:2]
                if role == "system":
                    messages.append(
                        {
                            "role": "system",
                            "content": r(content),
                        }
                    )
                elif role == "user":
                    if len(payload_item) >= 3:
                        # The user message contains an image
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": r(content)},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": r(payload_item[2])},
                                    },
                                ],
                            }
                        )
                    else:
                        # The user message contains text
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": r(content)},
                                ],
                            }
                        )
                elif role == "assistant":
                    # The assistant message contains a model response
                    messages.append(
                        {
                            "role": "assistant",
                            "content": r(content),
                        }
                    )
            response = self.model.infer_raw(messages)
            cache[conv["response_var_name"]] = response
            logger.debug(f"Sandbox cache[{conv['response_var_name']}] = {response}")
            if should_print_response:
                logger.info(f"Response to conversation #{i} ({conv['id']}):\n{response}")
        return cache["result"]

    def postprocess(self, result: str) -> int:
        # LLM provides a string as the output (response), so we need to convert it to an integer
        # We believe that the last appearance of `real` or `fake` or `generated` in the response is the final decision
        result = result.lower().strip()
        last_real = result.rfind("real")
        last_fake = max(result.rfind("fake"), result.rfind("generated"))
        # If both searches fail, we cannot determine the final decision
        if last_real == -1 and last_fake == -1:
            logger.warning("Could not determine the final decision from the response: {}", result)
            return -1
        return 1 if last_real > last_fake else 0

    def classify(self, sample: Path, label: int | bool = -1, should_print_response: bool = False) -> int:
        response = self.llm_query(sample, label, should_print_response)
        return self.postprocess(response)

    @staticmethod
    def _update_metrics(y_true: list, y_pred: list, pbar: tqdm) -> None:
        """Helper method to update progress bar with current metrics"""
        if y_pred:
            acc = accuracy_score(y_true, y_pred) * 100
            real_acc = (
                len([1 for i, y in enumerate(y_true) if y == 1 and y_pred[i] == 1]) / max(1, y_true.count(1)) * 100
            )
            fake_acc = (
                len([1 for i, y in enumerate(y_true) if y == 0 and y_pred[i] == 0]) / max(1, y_true.count(0)) * 100
            )
            pbar.set_postfix(
                all=f"{acc:.2f}%",
                reals=f"{real_acc:.2f}%",
                fakes=f"{fake_acc:.2f}%",
            )

    def evaluate(
        self,
        output_path: Path,
        continue_from: pd.DataFrame = None,
        should_print_response: bool = False,
    ) -> tuple[float, float, float]:
        if continue_from is not None:
            df = continue_from
            processed_samples = set(df["path"].apply(Path))
            self.samples = [(sample, label) for sample, label in self.samples if sample not in processed_samples]
            write_mode = "a"
            logger.info("Continuing evaluation with {} remaining samples", len(self.samples))
        else:
            df = pd.DataFrame(columns=["path", "label", "pred", "response"])
            write_mode = "w"
            logger.info(
                "Starting evaluation of {} samples ({} real, {} fake)",
                len(self.samples),
                len(self.real_samples),
                len(self.fake_samples),
            )

        y_true = []
        y_pred = []

        pbar = tqdm(enumerate(self.samples), total=len(self.samples), desc="Eval...")
        for i, (sample, label) in pbar:
            logger.debug("Processing sample {}/{}: {}", i, len(self.samples), sample.name)
            pbar.set_description(f"Eval {sample.name[:19]}")

            try:
                response = self.llm_query(sample, label, should_print_response)
                pred = self.postprocess(response)
                if pred != -1:
                    y_true.append(label)
                    y_pred.append(pred)
                else:
                    logger.warning("Invalid prediction for sample: {}", sample.name)

                new_row = pd.DataFrame(
                    {
                        "path": [sample],
                        "label": [label],
                        "pred": [pred],
                        "response": [response],
                    }
                )

                if i == 0 and write_mode == "w":
                    new_row.to_csv(output_path, index=False, mode="w")
                else:
                    new_row.to_csv(output_path, index=False, mode="a", header=False)

                self._update_metrics(y_true, y_pred, pbar)

            except Exception as e:
                logger.error("Error processing sample {}: {}", sample.name, e)
                continue

        if not y_pred:
            logger.error("No valid predictions were made during evaluation")
            return 0.0, 0.0, 0.0

        return self._calculate_final_metrics(y_true, y_pred)

    @staticmethod
    def _calculate_final_metrics(y_true: list, y_pred: list) -> tuple[float, float, float]:
        """Calculate and log final metrics"""
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            real_accuracy = len([1 for i, y in enumerate(y_true) if y == 1 and y_pred[i] == 1]) / max(
                1, y_true.count(1)
            )
            fake_accuracy = len([1 for i, y in enumerate(y_true) if y == 0 and y_pred[i] == 0]) / max(
                1, y_true.count(0)
            )

            logger.info(
                "Metrics\n"
                "- Overall Accuracy: {:.3f} %\n"
                "- Precision: {:.3f} %\n"
                "- Recall: {:.3f} %\n"
                "- Acc. on real images: {:.3f} %\n"
                "- Acc. on fake images: {:.3f} %",
                accuracy * 100,
                precision * 100,
                recall * 100,
                real_accuracy * 100,
                fake_accuracy * 100,
            )
            return accuracy, precision, recall
        except Exception as e:
            logger.error("Error calculating metrics: {}", e)
            return 0.0, 0.0, 0.0
