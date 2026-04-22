from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

from chmm import CHMM, collect_pair_codes_from_sequences


def dist_is_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def dist_barrier() -> None:
    if dist_is_enabled():
        dist.barrier()


def dist_all_reduce(tensor: torch.Tensor) -> None:
    if dist_is_enabled():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def shard_rows(seqs: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    if world_size <= 1:
        return seqs

    n = int(seqs.shape[0])
    rows_per_rank = ceil_div(n, world_size)
    start = rank * rows_per_rank
    end = min(n, start + rows_per_rank)
    if start >= n:
        return seqs[:0]
    return seqs[start:end]


def model_scalar_zeros(model: CHMM) -> torch.Tensor:
    param = next(iter(model.parameters()), None)
    if param is not None:
        return torch.zeros((), dtype=param.dtype, device=param.device)

    buf = next(iter(model.buffers()), None)
    if buf is not None:
        return torch.zeros((), dtype=buf.dtype, device=buf.device)

    return torch.zeros((), dtype=torch.float32, device="cpu")


def compute_loglikelihood(
    model: CHMM,
    data: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if data.shape[0] == 0:
        return model_scalar_zeros(model)
    return model.loglikelihood(data, batch_size)


def apply_dropout(
    input_ids: torch.Tensor,
    dropout: float,
    vocab_size: int,
    eos_token_id: int,
) -> torch.Tensor:
    del vocab_size
    if dropout <= 0.0:
        return input_ids

    n, d = input_ids.shape
    input_ids[torch.rand(n, device=input_ids.device) < dropout, -1] = eos_token_id
    random_mask = torch.rand(n, d, device=input_ids.device) < dropout
    input_ids[torch.logical_and(random_mask, input_ids != eos_token_id)] = -1
    return input_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--checkpoint", default=0, type=int)
    parser.add_argument("--save_per_step", default=10, type=int)

    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--total_chunks", required=True, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--sample_length", default=None, type=int)
    parser.add_argument("--em_schedule", required=True, type=str)

    parser.add_argument("--vocab_size", default=None, type=int)
    parser.add_argument("--eos_token_id", default=None, type=int)
    parser.add_argument("--tokenizer_name_or_path", default="", type=str)

    parser.add_argument("--clone_schedule_file", default="", type=str)
    parser.add_argument(
        "--clone_schedule_function",
        default="",
        type=str,
        help=(
            "Optional custom clone schedule builder as path:function_name. "
            "The path can point to a .py file or a .ipynb notebook."
        ),
    )
    parser.add_argument("--pair_code_chunk_count", default=0, type=int)

    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument(
        "--pseudocount",
        default=0.001,
        type=float,
        help="Per clone-to-clone transition kappa for paper-style pseudocount smoothing.",
    )
    parser.add_argument("--log_file", default="", type=str)

    parser.add_argument("--disable_mmap", action="store_true")
    parser.add_argument("--disable_pin_memory", action="store_true")
    parser.add_argument("--disable_tf32", action="store_true")

    return parser.parse_args()


def resolve_vocab_and_eos(args: argparse.Namespace) -> tuple[int, int]:
    if args.vocab_size is not None and args.eos_token_id is not None:
        return int(args.vocab_size), int(args.eos_token_id)

    if not args.tokenizer_name_or_path:
        raise ValueError(
            "Provide (--vocab_size and --eos_token_id) or --tokenizer_name_or_path."
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    return int(tokenizer.vocab_size), int(tokenizer.eos_token_id)


def chunk_file(data_path: str, dataset: str, chunk_idx: int, total_chunks: int) -> str:
    if total_chunks == 1:
        return f"{data_path}/{dataset}.train"
    return f"{data_path}/{dataset}.train.{chunk_idx}"


def parse_em_schedule(schedule: str) -> list[tuple[int, int]]:
    return [
        tuple(int(y) for y in item.split(","))
        for item in schedule.split(";")
        if item.strip()
    ]


def load_sequences(
    path: str,
    sample_length: int | None,
    *,
    mmap: bool = False,
) -> torch.Tensor:
    load_kwargs: dict[str, object] = {
        "map_location": "cpu",
        "weights_only": True,
    }
    if mmap:
        load_kwargs["mmap"] = True

    try:
        seqs = torch.load(path, **load_kwargs).long()
    except TypeError:
        load_kwargs.pop("mmap", None)
        seqs = torch.load(path, **load_kwargs).long()

    if sample_length is not None:
        seqs = seqs[:, :sample_length]
    return seqs.contiguous()


def count_token_frequency(seqs: torch.Tensor, vocab_size: int) -> torch.Tensor:
    flattened = seqs.reshape(-1)
    flattened = flattened[flattened >= 0]
    return torch.bincount(flattened, minlength=vocab_size)


def load_clone_schedule_function(spec: str):
    if not spec:
        raise ValueError(
            "--clone_schedule_function is required when initializing checkpoint-0. "
            "Pass a path:function_name that defines build_clone_schedule."
        )
    if ":" not in spec:
        raise ValueError(
            "--clone_schedule_function must look like path:function_name, "
            f"got {spec!r}."
        )

    path_text, function_name = spec.rsplit(":", 1)
    source_path = Path(path_text).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Clone schedule function path not found: {source_path}")

    if source_path.suffix == ".ipynb":
        with open(source_path, encoding="utf-8") as fin:
            notebook = json.load(fin)
        source_chunks = []
        function_header = f"def {function_name}("
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if function_header in source:
                source_chunks.append(source)
        if not source_chunks:
            raise ValueError(f"{function_name!r} was not found in {source_path}.")

        namespace = {"torch": torch}
        exec(compile("\n\n".join(source_chunks), str(source_path), "exec"), namespace)
        schedule_function = namespace.get(function_name)
    else:
        module_spec = importlib.util.spec_from_file_location(
            f"_custom_clone_schedule_{source_path.stem}",
            source_path,
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Could not import clone schedule module: {source_path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        schedule_function = getattr(module, function_name, None)

    if not callable(schedule_function):
        raise TypeError(f"{function_name!r} from {source_path} is not callable.")
    return schedule_function


def call_clone_schedule_function(
    schedule_function,
    args: argparse.Namespace,
    token_frequency: torch.Tensor,
    vocab_size: int,
    eos_token_id: int,
) -> torch.Tensor:
    kwargs = {
        "token_frequency": token_frequency,
        "vocab_size": vocab_size,
        "eos_token_id": eos_token_id,
        "args": args,
    }
    signature = inspect.signature(schedule_function)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_kwargs:
        result = schedule_function(**kwargs)
    else:
        accepted_names = {
            name
            for name, param in signature.parameters.items()
            if param.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
        result = schedule_function(
            **{name: value for name, value in kwargs.items() if name in accepted_names}
        )

    clones_per_token = torch.as_tensor(result, dtype=torch.long)
    if tuple(clones_per_token.shape) != (vocab_size,):
        raise ValueError(
            "Clone schedule function must return one clone count per token: "
            f"expected shape ({vocab_size},), got {tuple(clones_per_token.shape)}."
        )
    if torch.any(clones_per_token <= 0):
        raise ValueError("Clone schedule function must return positive clone counts.")
    return clones_per_token


def summarize_clone_schedule(clones_per_token: torch.Tensor) -> str:
    counts = torch.unique(clones_per_token, return_counts=True)
    parts = [f"{int(clone_count)}x:{int(num_tokens)}" for clone_count, num_tokens in zip(*counts)]
    hidden_states = int(clones_per_token.sum().item())
    return f"hidden_states={hidden_states}, schedule=({', '.join(parts)})"


def collect_pair_codes(
    args: argparse.Namespace,
    vocab_size: int,
) -> torch.Tensor:
    pair_code_set: set[int] = set()

    schedule_file = args.clone_schedule_file or f"{args.data_path}/{args.dataset}.lvd"
    if Path(schedule_file).exists():
        seqs = load_sequences(schedule_file, args.sample_length, mmap=not args.disable_mmap)
        pair_code_set.update(collect_pair_codes_from_sequences(seqs, vocab_size).tolist())

    num_chunks = (
        args.total_chunks
        if args.pair_code_chunk_count <= 0
        else min(args.pair_code_chunk_count, args.total_chunks)
    )

    for chunk_idx in range(num_chunks):
        path = chunk_file(args.data_path, args.dataset, chunk_idx, args.total_chunks)
        seqs = load_sequences(path, args.sample_length, mmap=not args.disable_mmap)
        pair_code_set.update(collect_pair_codes_from_sequences(seqs, vocab_size).tolist())

    if not pair_code_set:
        raise ValueError("No observed token pairs were found to initialize sparse CHMM blocks.")

    return torch.tensor(sorted(pair_code_set), dtype=torch.long)


def initialize_checkpoint_zero(
    args: argparse.Namespace,
    vocab_size: int,
    eos_token_id: int,
) -> CHMM:
    schedule_file = args.clone_schedule_file or f"{args.data_path}/{args.dataset}.lvd"
    if not Path(schedule_file).exists():
        raise FileNotFoundError(
            f"Clone schedule file not found: {schedule_file}. "
            "Generate sampled LVD sequences first or pass --clone_schedule_file."
        )

    print(f"load_sequences")
    seqs = load_sequences(schedule_file, args.sample_length, mmap=not args.disable_mmap)
    print(f"count_token_frequency")
    token_frequency = count_token_frequency(seqs, vocab_size)
    print(f"build_clone_schedule")
    schedule_function = load_clone_schedule_function(args.clone_schedule_function)
    clones_per_token = call_clone_schedule_function(
        schedule_function=schedule_function,
        args=args,
        token_frequency=token_frequency,
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
    )
    print(f"collect_pair_codes")
    pair_codes = collect_pair_codes(args, vocab_size)

    print(f"init model")
    model = CHMM(
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
        clones_per_token=clones_per_token,
        pair_codes=pair_codes,
    )

    print(f"save_pretrained")
    ckpt_dir = Path(args.model_path) / "checkpoint-0"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)

    print(f"!!Initialized CHMM checkpoint-0 from {schedule_file}!!")
    print(summarize_clone_schedule(clones_per_token))
    print(f"observed_pair_blocks={int(pair_codes.numel())}")

    return model


def maybe_initialize_checkpoint_zero(
    args: argparse.Namespace,
    rank: int,
    vocab_size: int,
    eos_token_id: int,
) -> CHMM | None:
    ckpt_dir = Path(args.model_path) / f"checkpoint-{args.checkpoint}"
    if ckpt_dir.exists():
        dist_barrier()
        return None

    if args.checkpoint != 0:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

    model = None
    if rank == 0:
        model = initialize_checkpoint_zero(args, vocab_size, eos_token_id)

    dist_barrier()
    return model


def configure_runtime(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    runtime: dict[str, object] = {
        "use_cuda": device.type == "cuda",
        "pin_memory": False,
        "non_blocking": False,
        "use_mmap_load": not args.disable_mmap,
        "allow_tf32": False,
        "gpu_name": "",
        "gpu_memory_gb": 0.0,
        "capability": None,
    }

    if device.type != "cuda":
        return runtime

    props = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    runtime["gpu_name"] = props.name
    runtime["gpu_memory_gb"] = props.total_memory / (1024 ** 3)
    runtime["capability"] = capability

    if not args.disable_pin_memory:
        runtime["pin_memory"] = True
        runtime["non_blocking"] = True

    is_ampere_or_newer = capability[0] >= 8
    if is_ampere_or_newer and not args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
        runtime["allow_tf32"] = True

    return runtime


def move_batch_to_device(
    cpu_batch: torch.Tensor,
    device: torch.device,
    runtime: dict[str, object],
) -> torch.Tensor:
    if device.type != "cuda":
        return cpu_batch

    batch = cpu_batch
    if runtime["pin_memory"]:
        batch = batch.pin_memory()
    return batch.to(device, non_blocking=bool(runtime["non_blocking"]))


def gather_train_eval_subset(
    eval_parts: list[torch.Tensor],
    target_rows: int,
    local_chunk: torch.Tensor,
) -> int:
    current_rows = sum(part.shape[0] for part in eval_parts)
    if current_rows >= target_rows:
        return current_rows

    take = min(target_rows - current_rows, local_chunk.shape[0])
    if take > 0:
        eval_parts.append(local_chunk[:take].clone())

    return current_rows + take


def train_chmm(
    rank: int,
    world_size: int,
    args: argparse.Namespace,
) -> None:
    print("----started-----")
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    runtime = configure_runtime(args, device)

    vocab_size, eos_token_id = resolve_vocab_and_eos(args)
    model = maybe_initialize_checkpoint_zero(args, rank, vocab_size, eos_token_id)

    if model is None:
        model = CHMM.from_pretrained(
            Path(args.model_path) / f"checkpoint-{args.checkpoint}",
            map_location="cpu",
        )

    model = model.to(device)
    model.eval()

    if rank == 0:
        if device.type == "cuda":
            print(
                "Runtime config: "
                f"gpu={runtime['gpu_name']}, "
                f"memory_gb={runtime['gpu_memory_gb']:.2f}, "
                f"capability={runtime['capability']}, "
                f"pin_memory={runtime['pin_memory']}, "
                f"non_blocking={runtime['non_blocking']}, "
                f"use_mmap_load={runtime['use_mmap_load']}, "
                f"allow_tf32={runtime['allow_tf32']}"
            )
        else:
            print(
                "Runtime config: "
                f"cpu_only, "
                f"use_mmap_load={runtime['use_mmap_load']}"
            )

    dev_all = load_sequences(
        f"{args.data_path}/{args.dataset}.dev",
        args.sample_length,
        mmap=bool(runtime["use_mmap_load"]),
    )
    dev_size = int(dev_all.shape[0])
    dev_data = shard_rows(dev_all, rank, world_size)

    em_schedule = parse_em_schedule(args.em_schedule)
    for _, step_size in em_schedule:
        if step_size > args.total_chunks:
            raise ValueError("Each EM schedule step_size must be <= total_chunks.")

    step_offset = args.checkpoint

    for step_count, step_size in em_schedule:
        for _ in range(step_count):
            if step_offset == args.checkpoint:
                with torch.inference_mode():
                    dev_ll = compute_loglikelihood(model, dev_data, args.batch_size)
                dist_all_reduce(dev_ll)

                if rank == 0:
                    msg = f"{args.checkpoint}\t{-1.0}\t{dev_ll.item() / max(dev_size, 1)}"
                    print(msg)
                    if args.log_file:
                        with open(args.log_file, "a+", encoding="utf-8") as fout:
                            fout.write(msg + "\n")

            transition_counts, gamma_counts = model.empty_count_buffers(device=device)
            train_eval_parts: list[torch.Tensor] = []
            train_eval_target_rows = int(dev_data.shape[0])

            with torch.inference_mode():
                for idx in range(step_offset, step_offset + step_size):
                    path = chunk_file(
                        args.data_path,
                        args.dataset,
                        idx % args.total_chunks,
                        args.total_chunks,
                    )
                    local_chunk = load_sequences(
                        path,
                        args.sample_length,
                        mmap=bool(runtime["use_mmap_load"]),
                    )
                    local_chunk = shard_rows(local_chunk, rank, world_size)

                    if local_chunk.shape[0] == 0:
                        continue
                    chunk_fully_observed = args.dropout <= 0.0 and not torch.any(local_chunk == -1)

                    gather_train_eval_subset(
                        train_eval_parts,
                        train_eval_target_rows,
                        local_chunk,
                    )

                    for batch_idx in tqdm(
                        range(0, local_chunk.shape[0], args.batch_size),
                        disable=rank != 0,
                    ):
                        cpu_batch = local_chunk[batch_idx : batch_idx + args.batch_size]
                        batch = move_batch_to_device(cpu_batch, device, runtime)
                        batch = apply_dropout(batch, args.dropout, vocab_size, eos_token_id)

                        if chunk_fully_observed or not torch.any(batch == -1):
                            model.accumulate_observed(batch, transition_counts, gamma_counts)
                        else:
                            probs = model.forward(batch)
                            model.backward(batch, probs, transition_counts, None, gamma_counts)

            dist_all_reduce(transition_counts)
            dist_all_reduce(gamma_counts)

            with torch.inference_mode():
                model.update_from_counts(
                    transition_counts=transition_counts,
                    gamma_counts=gamma_counts,
                    pseudocount=args.pseudocount,
                )

                if train_eval_parts:
                    train_data_eval = torch.cat(train_eval_parts, dim=0)
                else:
                    train_data_eval = dev_data[:0].clone()

                train_ll = compute_loglikelihood(model, train_data_eval, args.batch_size)
                dev_ll = compute_loglikelihood(model, dev_data, args.batch_size)

            dist_all_reduce(train_ll)
            dist_all_reduce(dev_ll)

            if rank == 0:
                ckpt = step_offset + step_size
                msg = f"{ckpt}\t{train_ll.item() / max(dev_size, 1)}\t{dev_ll.item() / max(dev_size, 1)}"
                print(msg)
                if args.log_file:
                    with open(args.log_file, "a+", encoding="utf-8") as fout:
                        fout.write(msg + "\n")

                if ckpt % args.save_per_step == 0 and ckpt != 0:
                    model.save_pretrained(Path(args.model_path) / f"checkpoint-{ckpt}")

            step_offset += step_size


if __name__ == "__main__":
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)

    Path(args.model_path).mkdir(parents=True, exist_ok=True)

    if rank == 0 and args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a+", encoding="utf-8") as fout:
            fout.write(str(vars(args)) + "\n")

    train_chmm(rank, world_size, args)
