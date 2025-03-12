import os
import random
import re
import sys
from pathlib import Path

import click
import orjson
import pandas as pd
from loguru import logger

import mllm_defake
from mllm_defake.classifiers.mllm_classifier import MLLMClassifier
from mllm_defake.datasets import SUPPORTED_DATASETS
from mllm_defake.finetune import SUPPORTED_CONFIGS
from mllm_defake.vllms import SUPPORTED_MODELS, VLLM


def find_prompt_file(prompt: str) -> dict:
    if Path(prompt).exists():
        p = Path(prompt)
        logger.info("Loading prompt file: {}", p)
        return orjson.loads(p.read_bytes())
    prompts_dir = Path("prompts")
    for path in prompts_dir.rglob("*.json"):
        if path.stem == prompt:
            logger.info("Assuming prompt file: {}", path)
            return orjson.loads(path.read_bytes())
    raise FileNotFoundError(f"Prompt file not found: {prompt}")


def load_model(model: str) -> VLLM:
    model = model.lower()
    if model == "gpt-4o" or model == "gpt4o":
        from mllm_defake.vllms import GPT4o

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for GPT-4o.")
        proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
        return GPT4o(api_key=api_key, proxy=proxy)
    elif model == "gpt-4o-mini" or model == "gpt4omini":
        from mllm_defake.vllms import GPT4oMini

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for GPT-4o-mini.")
        proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
        if proxy is None:
            logger.warning("Starting GPT-4o-mini inference without a proxy.")
        return GPT4oMini(api_key=api_key, proxy=proxy)
    elif model == "gpt-4.5-preview" or model == "gpt45":
        from mllm_defake.vllms import GPT45

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for GPT-4.5-preview.")
        proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
        if proxy is None:
            logger.warning("Starting GPT-4.5-preview inference without a proxy.")
        return GPT45(api_key=api_key, proxy=proxy)
    elif model == "llama-3.2-vision-instruct" or model == "llama32vi":
        from mllm_defake.vllms import Llama32VisionInstruct

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL") or "http://127.0.0.1:8000/v1"
        return Llama32VisionInstruct(api_key=api_key, base_url=base_url)
    elif model == "llama-3.2-vision-cot" or model == "llavacot":
        from mllm_defake.vllms import Llama32VisionCoT

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL") or "http://127.0.0.1:8000/v1"
        return Llama32VisionCoT(api_key=api_key, base_url=base_url)
    elif model == "qvq":
        from mllm_defake.vllms import QVQ

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL") or "http://127.0.0.1:8000/v1"
        return QVQ(api_key=api_key, base_url=base_url)
    elif model == "internvl-2.5" or model == "internvl25":
        from mllm_defake.vllms import InternVL25

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL") or "http://127.0.0.1:8000/v1"
        return InternVL25(api_key=api_key, base_url=base_url)
    elif model == "onevision":
        from mllm_defake.vllms import LLaVAOneVision

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL") or "http://127.0.0.1:8000/v1"
        return LLaVAOneVision(api_key=api_key, base_url=base_url)
    elif model == "qwen2vl":
        from mllm_defake.vllms import Qwen2VL

        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL") or "http://127.0.0.1:8000/v1"
        return Qwen2VL(api_key=api_key, base_url=base_url)
    else:
        raise ValueError(f"Invalid model: {model}")


def load_samples(
    dataset: str, count: int, real_dir: Path | None = None, fake_dir: Path | None = None
) -> tuple[list[Path], list[Path]]:
    if dataset == "WildFakeResampled":
        from mllm_defake.datasets import WildFakeResampled

        dataset = WildFakeResampled()
    elif dataset == "WildFakeResampled20K":
        from mllm_defake.datasets import WildFakeResampled

        dataset = WildFakeResampled("./WildFakeResampled20K")
    elif dataset == "ImageFolders":
        from mllm_defake.datasets import ImageFolders

        if not real_dir or not fake_dir:
            raise ValueError("If `--dataset` is empty, `--real_dir` and `--fake_dir` must be specified.")
        dataset = ImageFolders(real_dir, fake_dir)
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    real_samples, fake_samples = dataset.list_images()
    if count < len(real_samples):
        real_samples = random.sample(real_samples, count)
        logger.debug(f"Sampling {count} / {len(real_samples)} real samples from {dataset}")
    if count < len(fake_samples):
        fake_samples = random.sample(fake_samples, count)
        logger.debug(f"Sampling {count} / {len(fake_samples)} fake samples from {dataset}")
    logger.info(
        "Loaded {} real samples and {} fake samples from {}",
        len(real_samples),
        len(fake_samples),
        dataset,
    )
    return real_samples, fake_samples


@click.command()
@click.option(
    "-m",
    "--model",
    type=click.Choice(SUPPORTED_MODELS),
    help="The model to use for inference.",
    default="llama32vi",
)
@click.option(
    "-p",
    "--prompt",
    help="The name or path to the prompt v3 JSON file. Please refer to `prompts/readme.md` for format details.",
    default="simple",
)
@click.option(
    "-v",
    "--verbose",
    help="Verbose. Set if you would like to see every step of the classification process, including the model's full response.",
    is_flag=True,
)
@click.argument("image_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def classify(model: str, prompt: str, image_path: str, verbose: bool):
    """
    Classify a single image as real or fake using the specified model.

    IMAGE_PATH: Path to the image file to classify.
    """
    log_file = "logs/classify.log"
    logger.add(log_file, rotation="2 MB", backtrace=True, diagnose=True)
    try:
        # Load prompt & model
        prompt_config = find_prompt_file(prompt)
        model_instance = load_model(model)

        # Create classifier with single image
        image_path = Path(image_path)
        classifier = MLLMClassifier(prompt_config, model_instance, [], [])

        # Get prediction
        pred = classifier.classify(image_path, -1, should_print_response=verbose)

        # Map prediction to human-readable output
        if pred == -1:
            result = "unknown"
        elif pred == 1:
            result = "real"
        else:
            result = "fake"

        # Print result only (for easy command line usage)
        print(result)

    except Exception as e:
        logger.error(f"Error during classification: {e!s}")
        print("unknown")
        sys.exit(1)


@click.command()
@click.option(
    "-p",
    "--prompt",
    help="The name or path to the prompt v3 JSON file. Please refer to `prompts/readme.md` for format details. If a name is given, the corresponding JSON file should be placed under `prompts/` or its subdirectories. Note that the name of the prompt file must be unique, otherwise the first match will be used.",
    default="simple",
)
@click.option(
    "-m",
    "--model",
    type=click.Choice(SUPPORTED_MODELS),
    help="The model to use for inference.",
    default="llama32vi",
)
@click.option(
    "-d",
    "--dataset",
    type=click.Choice(SUPPORTED_DATASETS),
    help="The dataset to use for inference. If not specified, will read `--real_dir` and `--fake_dir`.",
    default="",
)
@click.option(
    "--real_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    help="The directory containing real images. Must be specified if `--dataset` is not provided.",
    default="",
)
@click.option(
    "--fake_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    help="The directory containing fake images. Must be specified if `--dataset` is not provided.",
    default="",
)
@click.option(
    "-c",
    "--count",
    type=int,
    help="Will evaluate on `count` real samples AND `count` fake samples. If the dataset has fewer samples than the count, all samples will be used. Otherwise, the samples will be randomly selected. Note that the number of real and fake samples will be equal, summing up to `2 * count` samples. Defaults to 10.",
    default=10,
)
@click.option(
    "-s",
    "--seed",
    type=int,
    help="The random seed to use for sample selection. Defaults to 3706.",
    default=3706,
)
@click.option(
    "-l",
    "--log_id",
    help="The human-readable ID of the log file to be generated. The log file will be saved under `logs/`.",
    default="",
)
@click.option(
    "-o",
    "--output",
    help="The name or path to the output CSV file. If a name is given, the CSV file will be saved under `outputs/`. If not specified, a sensible default name will be generated.",
    default="",
)
@click.option(
    "--continue",
    "continue_eval",
    is_flag=True,
    help="Continue evaluation from a previous run if the output file exists.",
)
@click.option(
    "--real_only",
    is_flag=True,
    help="If set, only evaluate on real samples.",
)
@click.option(
    "--fake_only",
    is_flag=True,
    help="If set, only evaluate on fake samples.",
)
@click.option(
    "--job_split",
    type=str,
    help="This parameter takes a `1/4`-like string to split the job into multiple parts. Index starts with zero and ends with `n` for `m/n`. This is meant to be used to spawn multiple jobs for the same dataset and prompt. Proceed with caution when pointing to the same output file, use with `-l` for custom log files.",
    default="",
)
@click.option(
    "-v",
    "--verbose",
    help="Verbose. Set if you would like to see every step of the classification process, including the model's full response.",
    is_flag=True,
)
def infer(
    prompt,
    model,
    dataset,
    count,
    log_id,
    output,
    continue_eval,
    real_dir,
    fake_dir,
    seed,
    real_only,
    fake_only,
    job_split,
    verbose,
):
    """
    This script evaluates the performance of a multimodal language model (MLLM) classifier on a dataset of real and fake images.

    The script uses a prompt file in JSON format to generate queries for the MLLM model. The model generates responses based on the queries, which are then used to classify the images as real or fake.

    It outputs a CSV file with the results of the classification.
    """
    random.seed(seed)
    # Setup paths to enable external decorators
    pwd = os.getcwd()
    if pwd not in sys.path:
        sys.path.append(pwd)

    # Create necessary directories
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = f"logs/{log_id}.log" if log_id else "logs/main.log"
    logger.add(log_file, rotation="2 MB", backtrace=True, diagnose=True)
    cli_call = " ".join(sys.argv)
    logger.debug("Hello world! Inference started with: {}", cli_call)

    # Load prompt
    try:
        prompt = find_prompt_file(prompt)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)
    except orjson.JSONDecodeError as e:
        logger.error("Invalid JSON found for prompt `{}`: {}", prompt, e)
        sys.exit(1)
    except Exception as e:
        logger.error("Error loading prompt `{}`: {}", prompt, e)
        sys.exit(1)

    # Determine output path
    if not output:
        readable_output_name = "{dataset}-{count}_{model}_{prompt}{log_id}.csv".format(
            dataset=dataset,
            count=count,
            model=model.short_name if hasattr(model, "short_name") else model,
            prompt=prompt["name"] if isinstance(prompt, dict) else Path(prompt).stem,
            log_id=f"_{log_id}" if log_id else "",
        )
        output_path = Path("outputs") / readable_output_name
    else:
        output_path = Path(output)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".csv")
        if not output_path.is_absolute():
            output_path = Path("outputs") / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Will save the output to {}", output_path.resolve())

    # Load model
    model = load_model(model)

    # Check for existing results if continuing
    continue_df = None
    if dataset == "" or dataset == "ImageFolders":
        if not real_dir or not fake_dir:
            raise ValueError("If `--dataset` is empty, `--real_dir` and `--fake_dir` must be specified.")
        dataset = "ImageFolders"
        real_samples, fake_samples = load_samples(dataset, count, real_dir, fake_dir)
    else:
        real_samples, fake_samples = load_samples(dataset, count)
    if continue_eval:
        if output_path.exists():
            continue_df = pd.read_csv(output_path)
            already_processed_reals: int = continue_df["label"].sum()
            already_processed_fakes: int = len(continue_df) - already_processed_reals
            processed_reals: list[Path] = continue_df[continue_df["label"] == 1]["path"].apply(Path).tolist()
            processed_fakes: list[Path] = continue_df[continue_df["label"] == 0]["path"].apply(Path).tolist()
            logger.info(
                "Continuing evaluation from {} ({} real and {} fake samples already processed)",
                output_path,
                already_processed_reals,
                already_processed_fakes,
            )
            reduce_count_reals = max(0, len(real_samples) - already_processed_reals)
            reduce_count_fakes = max(0, len(fake_samples) - already_processed_fakes)
            real_samples = [s for s in real_samples if s not in processed_reals][:reduce_count_reals]
            fake_samples = [s for s in fake_samples if s not in processed_fakes][:reduce_count_fakes]
            logger.info(
                "{} real and {} fake samples remaining to be processed",
                len(real_samples),
                len(fake_samples),
            )
        else:
            logger.warning("Output file not found for continuing evaluation, restarting from scratch.")
    else:
        if output_path.exists():
            logger.warning(
                "Output file already exists at {}. Use `--continue` to continue evaluation.",
                output_path,
            )
            input("Overwrite the file? Press Enter to continue, or Ctrl+C to exit.")
        logger.info("Starting evaluation from scratch.")

    if real_only and fake_only:
        logger.error("Cannot evaluate on both real and fake samples only.")
        sys.exit(1)
    if real_only:
        logger.info("Dropped fake samples for real-only evaluation.")
        fake_samples = []
    if fake_only:
        logger.info("Dropped real samples for fake-only evaluation.")
        real_samples = []

    if job_split is not None and job_split != "":
        m, n = map(int, job_split.split("/"))
        if m < 0 or n < 1 or m > n:
            raise ValueError("Invalid job split.")
        real_start_index, real_end_index = (
            len(real_samples) * (m - 1) // n,
            len(real_samples) * m // n,
        )
        fake_start_index, fake_end_index = (
            len(fake_samples) * (m - 1) // n,
            len(fake_samples) * m // n,
        )
        real_samples = real_samples[real_start_index:real_end_index]
        fake_samples = fake_samples[fake_start_index:fake_end_index]
        logger.info(
            "Job split {}/{} - Real samples: {}-{}, Fake samples: {}-{}",
            m,
            n,
            real_start_index,
            real_end_index,
            fake_start_index,
            fake_end_index,
        )

    classifier = MLLMClassifier(prompt, model, real_samples, fake_samples)
    if not output:
        readable_output_name = "{dataset}-{count}_{model}_{prompt}{log_id}.csv".format(
            dataset=dataset,
            count=count,
            model=model.short_name if hasattr(model, "short_name") else model,
            prompt=prompt["name"],
            log_id=f"_{log_id}" if log_id else "",
        )
        readable_output_path: Path = Path("outputs") / readable_output_name
    elif output.endswith(".csv"):
        readable_output_path = Path(output)
    else:
        readable_output_name = f"{output}.csv"
        readable_output_path = Path("outputs") / readable_output_name
    readable_output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.success("Starting evaluation with output: {}", readable_output_path)
    classifier.evaluate(
        output_path=f"outputs/{readable_output_name}",
        continue_from=continue_df,
        should_print_response=verbose,
    )


def guess_experiment_setup_from_path(path: Path) -> tuple[str, str, str, int]:
    name = path.stem

    # Check if the name is in the format of `dataset-count_model_prompt`
    # 1. match dataset from SUPPORTED_DATASETS
    def match_and_remove(full_str, sub_str) -> str:
        if full_str.startswith(sub_str + "-") or full_str.startswith(sub_str + "_"):
            return full_str[len(sub_str) + 1 :]
        return full_str

    dataset = ""
    for ds in SUPPORTED_DATASETS:
        new_name = match_and_remove(name, ds)
        if new_name != name:
            dataset = ds
            name = new_name
            break
    else:
        logger.warning("Could not determine dataset from the name: {}", name)
    # 2. match count
    count = 0
    count_str = re.search(r"^(\d+)_", name)
    if count_str:
        count = int(count_str.group(1))
        name = name[count_str.end() :]
    else:
        logger.warning("Could not determine count from the name: {}", name)
    # 3. match model from SUPPORTED_MODELS
    model = ""
    supported_models_sorted = sorted(SUPPORTED_MODELS, key=len, reverse=True)
    for m in supported_models_sorted:
        new_name = match_and_remove(name, m)
        if new_name != name:
            model = m
            name = new_name
            break
    else:
        logger.warning("Could not determine model from the name: {}", name)
    # 4. prompt is the remaining part
    prompt = name
    return dataset, model, prompt, count


def doc_writer(experiment_name: str) -> None:
    if Path(experiment_name).exists():
        df = pd.read_csv(experiment_name)
        real_experiment_name = Path(experiment_name).stem
    elif Path(f"outputs/{experiment_name}.csv").exists():
        df = pd.read_csv(f"outputs/{experiment_name}.csv")
        real_experiment_name = experiment_name
    else:
        raise FileNotFoundError(f"Experiment file not found for {experiment_name}")

    # Filter out failed predictions
    count_before = len(df)
    df = df[df["pred"] != -1]
    count_after = len(df)
    fails = count_before - count_after
    if count_before != count_after:
        logger.warning(
            "Filtered out {} failed predictions from the experiment",
            fails,
        )

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    accuracy = accuracy_score(df["label"], df["pred"])
    precision = precision_score(df["label"], df["pred"], zero_division=0)
    recall = recall_score(df["label"], df["pred"], zero_division=0)
    real_accuracy = len(df[(df["label"] == 1) & (df["pred"] == 1)]) / max(1, df["label"].sum())
    fake_accuracy = len(df[(df["label"] == 0) & (df["pred"] == 0)]) / max(1, (1 - df["label"]).sum())

    metric_str = (
        f"Metrics {real_experiment_name}\n"
        f" - Overall Accuracy: {accuracy * 100:.3f} %\n"
        f" - Precision: {precision * 100:.3f} %\n"
        f" - Recall: {recall * 100:.3f} %\n"
        f" - Acc. on real images: {real_accuracy * 100:.3f} %\n"
        f" - Acc. on fake images: {fake_accuracy * 100:.3f} %"
    )

    # Split dataframe into correct and wrong predictions
    df["correct"] = df["label"] == df["pred"]
    correct_df = df[df["correct"]]
    wrong_df = df[~df["correct"]]

    total_images = len(df)
    count_real = df["label"].sum()
    count_fake = total_images - count_real
    logger.info(metric_str)

    def write_table_rows(f, df_subset, markdown_dir):
        for _, row in df_subset.iterrows():
            image_name = Path(row["path"]).name
            image_path = Path(row["path"]).resolve()
            common_root = os.path.commonpath([image_path, markdown_dir])
            layers_up_to_common_root = len(markdown_dir.relative_to(common_root).parts)
            rel_to_image_path = image_path.relative_to(common_root)
            rel_path_str = "../" * layers_up_to_common_root + str(rel_to_image_path)
            response_edited = row["response"].replace("\n", "<br>").replace(" ", " ").replace("|", "\\|")
            gt = "gen'd" if row["label"] == 0 else "real"
            pred = "gen'd" if row["pred"] == 0 else "real"
            f.write(f"| ![Image {image_name}]({rel_path_str}) | {gt} | {pred} | {response_edited} |\n")

    with open(f"outputs/{real_experiment_name}.md", "w", encoding="utf-8") as f:
        f.write(f"# Experiment Results - `{real_experiment_name}`\n\n")
        f.write("## Metrics\n\n")
        f.write(f"- *{total_images} images, {count_real} real, {count_fake} generated.*\n\n")
        if fails > 0:
            f.write(f"- *{fails} failed predictions were filtered out.*\n\n")
        f.write(f"```\n{metric_str}\n```\n\n")

        # Write wrong predictions first
        f.write("## Wrong Predictions\n\n")
        f.write("| Image | Label | Prediction | Full Response |\n")
        f.write("| --- | --- | --- | --- |\n")
        markdown_dir = Path("outputs/").resolve()
        write_table_rows(f, wrong_df, markdown_dir)
        f.write("\n")

        # Write correct predictions
        f.write("## Correct Predictions\n\n")
        f.write("| Image | Label | Prediction | Full Response |\n")
        f.write("| --- | --- | --- | --- |\n")
        write_table_rows(f, correct_df, markdown_dir)
        f.write("\n")

        f.write(f"Created by {mllm_defake.__name__} - `v{mllm_defake.__version__}`")

    logger.success("Saved the experiment results to outputs/{}.md", real_experiment_name)
    d_, m_, p_, c_ = guess_experiment_setup_from_path(Path(real_experiment_name))
    return {
        "experiment_name": real_experiment_name,
        "dataset": d_,
        "model": m_,
        "prompt": p_,
        "count": c_,
        "output_path": f"outputs/{real_experiment_name}.md",
        "total_images": total_images,
        "count_real": count_real,
        "count_fake": count_fake,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "real_accuracy": real_accuracy,
        "fake_accuracy": fake_accuracy,
        "fails": fails,
    }


@click.command()
@click.argument("experiment_name", type=str, default="all")
@click.option(
    "-s",
    "--summarize_to",
    type=click.Path(file_okay=True, dir_okay=False),
    help="Summarize the results to a CSV file.",
    default=None,
)
def doc(experiment_name, summarize_to):
    """
    Prints the metrics for a given experiment, and then creates a markdown file that shows the image, label, prediction and model response in a tabular format.

    EXPERIMENT_NAME: The name of the experiment to compute the metrics for. You may use a CSV path instead of an experiment name. By default, the script will look for the CSV file under `outputs/`.
    """
    log_file = "logs/doc.log"
    logger.add(log_file, rotation="2 MB", backtrace=True, diagnose=True)
    if experiment_name == "all":
        res = {}
        for path in Path("outputs/").rglob("*.csv"):
            res[path.stem] = doc_writer(path)
        if summarize_to:
            df = pd.DataFrame.from_dict(res, orient="index")
            df.to_csv(summarize_to)
            logger.success("Summarized the results to {}", summarize_to)
        return
    if summarize_to:
        logger.warning(
            "Summarizing to a CSV file is not supported for documenting singular experiment. Ignoring the `summarize_to` option."
        )
    doc_writer(experiment_name)


@click.command()
@click.argument("config", type=str, default="")
@click.option(
    "-ls",
    "--list_configs",
    help="List all pre-defined supported finetuning configurations.",
    is_flag=True,
)
def finetune(config, list_configs):
    """
    Finetune the model using the specified configuration.

    CONFIG: local file in yml or yaml format, or one of the pre-defined supported configurations.
    """
    if list_configs:
        logger.info("Pre-defined supported finetuning configurations:")
        for k in SUPPORTED_CONFIGS:
            logger.info(f" - {k}")
        return
    # check if config is a pre-defined config or a existing file
    if not (config in SUPPORTED_CONFIGS or Path(config).exists()):
        logger.error(f"Invalid config: {config}")
        return
    config = SUPPORTED_CONFIGS.get(config, config)
    # get train mode
    train_mode = config.split("/")[-1].split("_")[0]
    if train_mode == "sft":
        from mllm_defake.finetune.sft import sft_train

        sft_train(config)
    elif train_mode == "grpo":
        from mllm_defake.finetune.grpo import grpo_train

        grpo_train(config)
    else:
        logger.error(
            f"Invalid train mode: {train_mode}"
            "train model is defined by the first part of the config file name, e.g. sft_qwen2_5_vl_3b.yml. "
            "supported train modes: sft, grpo"
        )


@click.command()
@click.argument("lora_path", type=str, default="")
@click.option(
    "-o",
    "--output_path",
    help="The path to save the merged model.",
    default="",
)
def merge_lora(lora_path, output_path):
    """
    Merge the LoRA model with the base model.
    """
    from swift.llm import ExportArguments, merge_lora

    if lora_path.endswith("/"):
        lora_path = lora_path[:-1]
    if output_path == "":
        output_path = f"{lora_path}-merged"

    merge_lora(
        ExportArguments(
            adapters=[lora_path],
            merge_lora=True,
            output_dir=output_path,
        ),
        device_map="cpu",
    )


@click.group()
def main_cli():
    pass


main_cli.add_command(classify)
main_cli.add_command(infer)
main_cli.add_command(doc)
main_cli.add_command(finetune)
main_cli.add_command(merge_lora)


if __name__ == "__main__":
    main_cli()
