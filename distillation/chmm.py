from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

__all__ = ["CHMM", "collect_pair_codes_from_sequences"]


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)


def ib_ib_bj_to_ij(
    pf: torch.Tensor,
    pp: torch.Tensor,
    cp: torch.Tensor,
) -> torch.Tensor:
    ll = torch.amax(cp, dim=-1)
    ll = torch.where(torch.isfinite(ll), ll, torch.zeros_like(ll))

    pp = torch.exp(pp - ll[None, :])
    cp = torch.exp(cp - ll[:, None])

    pp = torch.nan_to_num(pp, nan=0.0, posinf=0.0, neginf=0.0)
    cp = torch.nan_to_num(cp, nan=0.0, posinf=0.0, neginf=0.0)

    ratio = pf / pp
    ratio[pp == 0.0] = 0.0
    ratio = torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

    return matmul(ratio, cp)


def collect_pair_codes_from_sequences(
    input_ids: torch.Tensor | Sequence[Sequence[int]],
    vocab_size: int,
) -> torch.Tensor:
    input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    if input_ids.shape[1] < 2:
        return torch.empty(0, dtype=torch.long)

    src = input_ids[:, :-1]
    dst = input_ids[:, 1:]
    observed = (src != -1) & (dst != -1)
    if not observed.any():
        return torch.empty(0, dtype=torch.long)

    pair_codes = src[observed] * vocab_size + dst[observed]
    return torch.unique(pair_codes, sorted=True).cpu()


@dataclass(frozen=True)
class BlockSpec:
    src_token: int
    dst_token: int
    src_start: int
    dst_start: int
    src_size: int
    dst_size: int
    value_start: int
    value_stop: int

    @property
    def src_stop(self) -> int:
        return self.src_start + self.src_size

    @property
    def dst_stop(self) -> int:
        return self.dst_start + self.dst_size

    @property
    def pair_code(self) -> int:
        raise RuntimeError("pair_code is stored externally, not in BlockSpec.")


class SparseTransitionTable(nn.Module):
    def __init__(
        self,
        clones_per_token: torch.Tensor,
        pair_codes: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float32,
        init_random: bool = True,
    ) -> None:
        super().__init__()

        clones_per_token = torch.as_tensor(clones_per_token, dtype=torch.long)
        pair_codes = self._normalize_pair_codes(pair_codes, len(clones_per_token))
        specs = self._build_block_specs(clones_per_token, pair_codes)
        values = (
            self._random_transition_values(clones_per_token, specs, dtype)
            if init_random
            else torch.empty(specs[-1].value_stop if specs else 0, dtype=dtype)
        )

        self.register_buffer("pair_codes", pair_codes)
        self.register_buffer("transition_values", values)
        self.register_buffer(
            "block_value_start",
            torch.tensor([spec.value_start for spec in specs], dtype=torch.long),
        )
        self.register_buffer(
            "block_src_start",
            torch.tensor([spec.src_start for spec in specs], dtype=torch.long),
        )
        self.register_buffer(
            "block_dst_start",
            torch.tensor([spec.dst_start for spec in specs], dtype=torch.long),
        )
        self.register_buffer(
            "block_src_size",
            torch.tensor([spec.src_size for spec in specs], dtype=torch.long),
        )
        self.register_buffer(
            "block_dst_size",
            torch.tensor([spec.dst_size for spec in specs], dtype=torch.long),
        )

        self.block_specs = specs
        self.outgoing_block_ids = self._build_outgoing_lists(len(clones_per_token), specs)
        self.incoming_block_ids = self._build_incoming_lists(len(clones_per_token), specs)

    def block_index_for_pair_code(self, pair_code: int) -> int | None:
        if self.pair_codes.numel() == 0:
            return None

        query = torch.tensor([pair_code], device=self.pair_codes.device, dtype=self.pair_codes.dtype)
        position = int(torch.searchsorted(self.pair_codes, query).item())
        if position >= int(self.pair_codes.numel()):
            return None
        if int(self.pair_codes[position].item()) != pair_code:
            return None
        return position

    @staticmethod
    def _normalize_pair_codes(
        pair_codes: torch.Tensor | Sequence[int] | Sequence[tuple[int, int]] | None,
        vocab_size: int,
    ) -> torch.Tensor:
        if pair_codes is None:
            if vocab_size > 512:
                raise ValueError(
                    "pair_codes is required for large sparse CHMMs. "
                    "Use collect_pair_codes_from_sequences(...) first."
                )
            pair_codes = torch.arange(vocab_size * vocab_size, dtype=torch.long)
        elif torch.is_tensor(pair_codes):
            pair_codes = pair_codes.to(dtype=torch.long, device="cpu")
        else:
            first_item = next(iter(pair_codes), None)
            if first_item is None:
                pair_codes = torch.empty(0, dtype=torch.long)
            elif isinstance(first_item, tuple):
                pair_codes = torch.tensor(
                    [src * vocab_size + dst for src, dst in pair_codes],
                    dtype=torch.long,
                )
            else:
                pair_codes = torch.tensor(list(pair_codes), dtype=torch.long)

        if pair_codes.numel() == 0:
            return pair_codes

        pair_codes = torch.unique(pair_codes, sorted=True)
        if torch.any(pair_codes < 0) or torch.any(pair_codes >= vocab_size * vocab_size):
            raise ValueError("pair_codes contains an out-of-range token pair.")
        return pair_codes

    @staticmethod
    def _build_block_specs(
        clones_per_token: torch.Tensor,
        pair_codes: torch.Tensor,
    ) -> list[BlockSpec]:
        state_offsets = torch.zeros(len(clones_per_token) + 1, dtype=torch.long)
        state_offsets[1:] = torch.cumsum(clones_per_token, dim=0)

        specs: list[BlockSpec] = []
        value_cursor = 0
        vocab_size = len(clones_per_token)
        for code in pair_codes.tolist():
            src_token = code // vocab_size
            dst_token = code % vocab_size

            src_start = int(state_offsets[src_token].item())
            dst_start = int(state_offsets[dst_token].item())
            src_size = int(clones_per_token[src_token].item())
            dst_size = int(clones_per_token[dst_token].item())
            value_count = src_size * dst_size

            specs.append(
                BlockSpec(
                    src_token=src_token,
                    dst_token=dst_token,
                    src_start=src_start,
                    dst_start=dst_start,
                    src_size=src_size,
                    dst_size=dst_size,
                    value_start=value_cursor,
                    value_stop=value_cursor + value_count,
                )
            )
            value_cursor += value_count
        return specs

    @staticmethod
    def _build_outgoing_lists(vocab_size: int, specs: Sequence[BlockSpec]) -> list[list[int]]:
        outgoing = [[] for _ in range(vocab_size)]
        for idx, spec in enumerate(specs):
            outgoing[spec.src_token].append(idx)
        return outgoing

    @staticmethod
    def _build_incoming_lists(vocab_size: int, specs: Sequence[BlockSpec]) -> list[list[int]]:
        incoming = [[] for _ in range(vocab_size)]
        for idx, spec in enumerate(specs):
            incoming[spec.dst_token].append(idx)
        return incoming

    def _random_transition_values(
        self,
        clones_per_token: torch.Tensor,
        specs: Sequence[BlockSpec],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not specs:
            return torch.empty(0, dtype=dtype)

        values = torch.empty(specs[-1].value_stop, dtype=dtype)
        outgoing = self._build_outgoing_lists(len(clones_per_token), specs)

        for src_token, block_ids in enumerate(outgoing):
            if not block_ids:
                continue

            src_size = int(clones_per_token[src_token].item())
            total_dst = sum(specs[block_idx].dst_size for block_idx in block_ids)
            packed = torch.rand(src_size, total_dst, dtype=dtype)
            packed /= packed.sum(dim=1, keepdim=True)

            cursor = 0
            for block_idx in block_ids:
                spec = specs[block_idx]
                width = spec.dst_size
                self.block_view(values, spec)[:, :] = packed[:, cursor : cursor + width]
                cursor += width

        return values

    def block_view(self, flat_values: torch.Tensor, spec: BlockSpec) -> torch.Tensor:
        return flat_values[spec.value_start : spec.value_stop].view(spec.src_size, spec.dst_size)

    def block(self, block_idx: int) -> torch.Tensor:
        return self.block_view(self.transition_values, self.block_specs[block_idx])

    def empty_counts(self, device: torch.device | None = None) -> torch.Tensor:
        device = self.transition_values.device if device is None else device
        return torch.zeros(
            self.transition_values.shape[0],
            device=device,
            dtype=self.transition_values.dtype,
        )

    def update_from_dense(self, dense_transition: torch.Tensor) -> None:
        new_values = torch.zeros_like(self.transition_values)
        for spec in self.block_specs:
            dense_block = dense_transition[
                spec.src_start : spec.src_stop,
                spec.dst_start : spec.dst_stop,
            ]
            self.block_view(new_values, spec).copy_(dense_block)
        self.transition_values.copy_(new_values)

    def normalized_values_from_counts(
        self,
        transition_counts: torch.Tensor,
        pseudocount: float,
        hidden_states: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pseudocount < 0.0:
            raise ValueError("pseudocount must be non-negative.")

        transition_counts = transition_counts.to(
            device=self.transition_values.device,
            dtype=self.transition_values.dtype,
        )
        new_values = torch.zeros_like(self.transition_values)
        row_totals = torch.zeros(
            hidden_states,
            device=self.transition_values.device,
            dtype=self.transition_values.dtype,
        )

        group_codes = None
        size_base = 0
        if self.transition_values.numel() > 0:
            size_base = int(self.block_dst_size.max().item()) + 1
            group_codes = self.block_src_size * size_base + self.block_dst_size
            for code in torch.unique(group_codes):
                src_size = int((code // size_base).item())
                dst_size = int((code % size_base).item())
                block_ids = (group_codes == code).nonzero(as_tuple=False).squeeze(-1)
                starts = self.block_value_start[block_ids]
                value_offsets = starts[:, None] + torch.arange(
                    src_size * dst_size,
                    device=self.transition_values.device,
                )
                block_counts = transition_counts[value_offsets].view(-1, src_size, dst_size)
                row_counts = block_counts.sum(dim=2)
                row_offsets = self.block_src_start[block_ids, None] + torch.arange(
                    src_size,
                    device=self.transition_values.device,
                )
                row_totals.index_add_(0, row_offsets.reshape(-1), row_counts.reshape(-1))

        denom = row_totals + float(pseudocount) * hidden_states
        valid_rows = denom > 0.0
        safe_denom = torch.where(valid_rows, denom, torch.ones_like(denom))

        if group_codes is not None:
            for code in torch.unique(group_codes):
                src_size = int((code // size_base).item())
                dst_size = int((code % size_base).item())
                block_ids = (group_codes == code).nonzero(as_tuple=False).squeeze(-1)
                starts = self.block_value_start[block_ids]
                value_offsets = starts[:, None] + torch.arange(
                    src_size * dst_size,
                    device=self.transition_values.device,
                )
                block_counts = transition_counts[value_offsets].view(-1, src_size, dst_size)
                row_offsets = self.block_src_start[block_ids, None] + torch.arange(
                    src_size,
                    device=self.transition_values.device,
                )
                block_denom = safe_denom[row_offsets].unsqueeze(-1)
                new_values[value_offsets.reshape(-1)] = (block_counts / block_denom).reshape(-1)

        floor = torch.zeros_like(row_totals)
        if pseudocount > 0.0:
            floor = torch.where(
                valid_rows,
                torch.full_like(row_totals, float(pseudocount)) / safe_denom,
                floor,
            )

        return new_values, floor

    def to_dense(self, hidden_states: int) -> torch.Tensor:
        dense = torch.zeros(
            hidden_states,
            hidden_states,
            device=self.transition_values.device,
            dtype=self.transition_values.dtype,
        )
        for block_idx, spec in enumerate(self.block_specs):
            dense[
                spec.src_start : spec.src_stop,
                spec.dst_start : spec.dst_stop,
            ] = self.block(block_idx)
        return dense


class CHMM(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        vocab_size: int,
        eos_token_id: int,
        clones_per_token: list[int] | torch.Tensor,
        *,
        pair_codes: torch.Tensor | Sequence[int] | Sequence[tuple[int, int]] | None = None,
        dtype: torch.dtype = torch.float32,
        init_random: bool = True,
    ) -> None:
        super().__init__()

        clones_per_token = torch.as_tensor(clones_per_token, dtype=torch.long)
        if len(clones_per_token) != vocab_size:
            raise ValueError("One clone count is required for each token.")

        hidden_states = int(clones_per_token.sum().item())
        gamma = torch.log_softmax(torch.randn(hidden_states, dtype=dtype), dim=0)
        state_offsets = torch.zeros(vocab_size + 1, dtype=torch.long)
        state_offsets[1:] = torch.cumsum(clones_per_token, dim=0)

        self.register_buffer("clones_per_token", clones_per_token)
        self.register_buffer(
            "clone_to_token",
            torch.repeat_interleave(torch.arange(vocab_size, dtype=torch.long), clones_per_token),
        )
        self.register_buffer("state_offsets", state_offsets)
        self.register_buffer("gamma", gamma)
        self.register_buffer("transition_floor", torch.zeros(hidden_states, dtype=dtype))

        self.transitions = SparseTransitionTable(
            clones_per_token=clones_per_token,
            pair_codes=pair_codes,
            dtype=dtype,
            init_random=init_random,
        )

        self.hidden_states = hidden_states
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.max_clones = int(clones_per_token.max().item())
        self.clone_size_values = sorted({int(size) for size in clones_per_token.tolist()})

    @property
    def device(self) -> torch.device:
        return self.gamma.device

    @property
    def pair_codes(self) -> torch.Tensor:
        return self.transitions.pair_codes

    @property
    def alpha_exp(self) -> torch.Tensor:
        dense = self.transitions.to_dense(self.hidden_states)
        return dense + self.transition_floor.unsqueeze(1)

    def config_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "clones_per_token": self.clones_per_token.tolist(),
            "num_pair_blocks": int(self.pair_codes.numel()),
        }

    @torch.no_grad()
    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": self.config_dict(),
            "gamma": self.gamma.detach().cpu(),
            "pair_codes": self.pair_codes.detach().cpu(),
            "transition_values": self.transitions.transition_values.detach().cpu(),
            "transition_floor": self.transition_floor.detach().cpu(),
        }
        torch.save(payload, output_dir / "model.pt")

        with open(output_dir / "config.json", "w", encoding="utf-8") as fout:
            json.dump(payload["config"], fout, indent=2)

    @classmethod
    def from_pretrained(cls, model_path: str | Path, map_location: str | torch.device | None = None) -> "CHMM":
        model_path = Path(model_path)
        payload = torch.load(model_path / "model.pt", map_location="cpu", weights_only=True)

        pair_codes = payload.get("pair_codes")
        if pair_codes is None and "alpha_exp" in payload:
            pair_codes = cls._pair_codes_from_dense(
                alpha_exp=payload["alpha_exp"],
                clones_per_token=torch.tensor(payload["config"]["clones_per_token"], dtype=torch.long),
                vocab_size=payload["config"]["vocab_size"],
            )

        model = cls(
            vocab_size=payload["config"]["vocab_size"],
            eos_token_id=payload["config"]["eos_token_id"],
            clones_per_token=payload["config"]["clones_per_token"],
            pair_codes=pair_codes,
            dtype=payload["gamma"].dtype,
            init_random=False,
        )

        if "transition_values" in payload:
            model.update_params(payload["transition_values"], payload["gamma"])
        else:
            model.update_params(payload["alpha_exp"], payload["gamma"])

        if "transition_floor" in payload:
            model.transition_floor.copy_(payload["transition_floor"].to(dtype=model.gamma.dtype))

        if map_location is not None:
            model = model.to(map_location)
        return model

    @staticmethod
    def _pair_codes_from_dense(
        alpha_exp: torch.Tensor,
        clones_per_token: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        state_offsets = torch.zeros(vocab_size + 1, dtype=torch.long)
        state_offsets[1:] = torch.cumsum(clones_per_token, dim=0)
        pair_codes = []
        for src_token in range(vocab_size):
            for dst_token in range(vocab_size):
                src_slice = slice(int(state_offsets[src_token]), int(state_offsets[src_token + 1]))
                dst_slice = slice(int(state_offsets[dst_token]), int(state_offsets[dst_token + 1]))
                if alpha_exp[src_slice, dst_slice].abs().sum().item() > 0:
                    pair_codes.append(src_token * vocab_size + dst_token)
        return torch.tensor(pair_codes, dtype=torch.long)

    @torch.no_grad()
    def update_params(
        self,
        transition_values: torch.Tensor,
        gamma: torch.Tensor,
        transition_floor: torch.Tensor | None = None,
    ) -> None:
        gamma = gamma.to(self.gamma.device, dtype=self.gamma.dtype)
        self.gamma.copy_(gamma)

        transition_values = transition_values.to(
            self.transitions.transition_values.device,
            dtype=self.transitions.transition_values.dtype,
        )
        if transition_values.ndim == 2:
            self.transitions.update_from_dense(transition_values)
        elif transition_values.ndim == 1:
            if transition_values.shape != self.transitions.transition_values.shape:
                raise ValueError("transition_values has the wrong sparse storage shape.")
            self.transitions.transition_values.copy_(transition_values)
        else:
            raise ValueError("transition_values must be either dense [H, H] or sparse flat [nnz].")

        if transition_floor is None:
            self.transition_floor.zero_()
        else:
            transition_floor = transition_floor.to(self.device, dtype=self.gamma.dtype)
            if transition_floor.shape != self.transition_floor.shape:
                raise ValueError("transition_floor has the wrong shape.")
            self.transition_floor.copy_(transition_floor)

    def empty_count_buffers(
        self,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.device if device is None else device
        transition_counts = self.transitions.empty_counts(device=device)
        gamma_counts = torch.zeros(self.hidden_states, device=device, dtype=self.gamma.dtype)
        return transition_counts, gamma_counts

    @torch.no_grad()
    def update_from_counts(
        self,
        transition_counts: torch.Tensor,
        gamma_counts: torch.Tensor,
        pseudocount: float,
    ) -> None:
        new_transition_values, new_transition_floor = self.transitions.normalized_values_from_counts(
            transition_counts=transition_counts,
            pseudocount=pseudocount,
            hidden_states=self.hidden_states,
        )
        gamma_counts = gamma_counts.to(self.device, dtype=self.gamma.dtype)
        gamma_counts = gamma_counts + pseudocount / self.hidden_states
        gamma_counts = gamma_counts / gamma_counts.sum()
        self.update_params(new_transition_values, torch.log(gamma_counts), new_transition_floor)

    def _token_slice(self, token_id: int) -> slice:
        start = int(self.state_offsets[token_id].item())
        stop = int(self.state_offsets[token_id + 1].item())
        return slice(start, stop)

    def _effective_block(self, block_idx: int) -> torch.Tensor:
        spec = self.transitions.block_specs[block_idx]
        floor = self.transition_floor[spec.src_start : spec.src_stop].unsqueeze(1)
        return self.transitions.block(block_idx) + floor

    def _floor_block(self, source_slice: slice, dest_slice: slice) -> torch.Tensor:
        src_size = source_slice.stop - source_slice.start
        dst_size = dest_slice.stop - dest_slice.start
        floor = self.transition_floor[source_slice].unsqueeze(1)
        return floor.expand(src_size, dst_size)

    def _floor_project(self, source_slice: slice, child_messages: torch.Tensor) -> torch.Tensor:
        floor = torch.log(self.transition_floor[source_slice]).unsqueeze(1)
        child_logsum = torch.logsumexp(child_messages, dim=0, keepdim=True)
        return floor + child_logsum

    def _stable_project(self, transition: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        msg_max = torch.amax(messages, dim=0, keepdim=True)
        msg_max = torch.where(torch.isfinite(msg_max), msg_max, torch.zeros_like(msg_max))
        probs = torch.exp(messages - msg_max)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        out = matmul(transition, probs)
        out = torch.log(out) + msg_max
        return out

    def _observed_pair_block_indices(
        self,
        curr_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair_codes = curr_tokens * self.vocab_size + next_tokens
        num_blocks = int(self.transitions.pair_codes.numel())
        if num_blocks == 0:
            return torch.zeros_like(pair_codes), torch.zeros_like(pair_codes, dtype=torch.bool)

        positions = torch.searchsorted(self.transitions.pair_codes, pair_codes)
        in_range = positions < num_blocks
        block_idx = torch.clamp(positions, max=num_blocks - 1)
        matched = in_range & (self.transitions.pair_codes[block_idx] == pair_codes)
        return block_idx, matched

    def _compact_size_groups(
        self,
        src_sizes: torch.Tensor,
        dst_sizes: torch.Tensor,
    ) -> Iterable[tuple[int, int, torch.Tensor]]:
        for src_size in self.clone_size_values:
            src_mask = src_sizes == src_size
            for dst_size in self.clone_size_values:
                cols = (src_mask & (dst_sizes == dst_size)).nonzero(as_tuple=False).squeeze(-1)
                if cols.numel() > 0:
                    yield src_size, dst_size, cols

    def _compact_token_size_groups(
        self,
        token_sizes: torch.Tensor,
    ) -> Iterable[tuple[int, torch.Tensor]]:
        for size in self.clone_size_values:
            cols = (token_sizes == size).nonzero(as_tuple=False).squeeze(-1)
            if cols.numel() > 0:
                yield size, cols

    def _compact_effective_blocks(
        self,
        curr_tokens: torch.Tensor,
        cols: torch.Tensor,
        block_idx: torch.Tensor,
        matched: torch.Tensor,
        src_size: int,
        dst_size: int,
    ) -> torch.Tensor:
        device = curr_tokens.device
        dtype = self.transitions.transition_values.dtype
        blocks = torch.zeros(
            (cols.shape[0], src_size, dst_size),
            device=device,
            dtype=dtype,
        )

        local_matched = matched[cols]
        if local_matched.any():
            local_rows = local_matched.nonzero(as_tuple=False).squeeze(-1)
            local_block_idx = block_idx[cols[local_rows]]
            starts = self.transitions.block_value_start[local_block_idx]
            offsets = starts[:, None] + torch.arange(src_size * dst_size, device=device)
            values = self.transitions.transition_values[offsets].view(-1, src_size, dst_size)
            blocks[local_rows] = values

        src_starts = self.state_offsets[curr_tokens[cols]]
        src_offsets = src_starts[:, None] + torch.arange(src_size, device=device)
        floor = self.transition_floor[src_offsets]
        return blocks + floor.unsqueeze(-1)

    def _compact_project_observed_step(
        self,
        curr_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
        child_messages: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = curr_tokens.shape[0]
        parent_messages = torch.full(
            (batch_size, self.max_clones),
            float("-inf"),
            device=self.device,
            dtype=self.gamma.dtype,
        )
        src_sizes = self.clones_per_token[curr_tokens]
        dst_sizes = self.clones_per_token[next_tokens]
        block_idx, matched = self._observed_pair_block_indices(curr_tokens, next_tokens)

        for src_size, dst_size, cols in self._compact_size_groups(src_sizes, dst_sizes):
            blocks = self._compact_effective_blocks(
                curr_tokens=curr_tokens,
                cols=cols,
                block_idx=block_idx,
                matched=matched,
                src_size=src_size,
                dst_size=dst_size,
            )
            child = child_messages[cols, :dst_size]
            child_max = torch.amax(child, dim=1, keepdim=True)
            child_max = torch.where(torch.isfinite(child_max), child_max, torch.zeros_like(child_max))
            child_probs = torch.exp(child - child_max)
            child_probs = torch.nan_to_num(child_probs, nan=0.0, posinf=0.0, neginf=0.0)
            projected = torch.bmm(blocks, child_probs.unsqueeze(-1)).squeeze(-1)
            parent_messages[cols, :src_size] = torch.log(projected) + child_max

        return parent_messages

    def forward_observed_compact(self, input_ids: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        if torch.any(input_ids == -1):
            raise ValueError("forward_observed_compact requires fully observed input_ids.")

        batch_size, seq_len = input_ids.shape
        messages: list[torch.Tensor] = [
            torch.empty(0, device=self.device, dtype=self.gamma.dtype)
            for _ in range(seq_len)
        ]

        last_tokens = input_ids[:, -1]
        last_sizes = self.clones_per_token[last_tokens]
        last_messages = torch.full(
            (batch_size, self.max_clones),
            float("-inf"),
            device=self.device,
            dtype=self.gamma.dtype,
        )
        for size, cols in self._compact_token_size_groups(last_sizes):
            last_messages[cols, :size] = 0.0
        messages[-1] = last_messages

        for t in range(seq_len - 2, -1, -1):
            messages[t] = self._compact_project_observed_step(
                curr_tokens=input_ids[:, t],
                next_tokens=input_ids[:, t + 1],
                child_messages=messages[t + 1],
            )

        first_tokens = input_ids[:, 0]
        first_sizes = self.clones_per_token[first_tokens]
        first_starts = self.state_offsets[first_tokens]
        loglik = torch.full(
            (batch_size,),
            float("-inf"),
            device=self.device,
            dtype=self.gamma.dtype,
        )
        for size, cols in self._compact_token_size_groups(first_sizes):
            offsets = first_starts[cols, None] + torch.arange(size, device=self.device)
            loglik[cols] = torch.logsumexp(self.gamma[offsets] + messages[0][cols, :size], dim=1)

        return messages, loglik

    def backward_observed_compact(
        self,
        input_ids: torch.Tensor,
        messages: list[torch.Tensor],
        loglik: torch.Tensor,
        transition_counts: torch.Tensor,
        gamma_flow: torch.Tensor,
    ) -> None:
        if torch.any(input_ids == -1):
            raise ValueError("backward_observed_compact requires fully observed input_ids.")

        batch_size, seq_len = input_ids.shape
        pf = torch.zeros(
            (batch_size, self.max_clones),
            device=self.device,
            dtype=self.gamma.dtype,
        )

        first_tokens = input_ids[:, 0]
        first_sizes = self.clones_per_token[first_tokens]
        first_starts = self.state_offsets[first_tokens]
        for size, cols in self._compact_token_size_groups(first_sizes):
            offsets = first_starts[cols, None] + torch.arange(size, device=self.device)
            local_pf = torch.exp(self.gamma[offsets] + messages[0][cols, :size] - loglik[cols, None])
            local_pf = torch.nan_to_num(local_pf, nan=0.0, posinf=0.0, neginf=0.0)
            pf[cols, :size] = local_pf
            gamma_flow.index_add_(0, offsets.reshape(-1), local_pf.reshape(-1))

        for t in range(seq_len - 1):
            pf = self._compact_accumulate_observed_step(
                curr_tokens=input_ids[:, t],
                next_tokens=input_ids[:, t + 1],
                parent_messages=messages[t],
                child_messages=messages[t + 1],
                pf=pf,
                transition_counts=transition_counts,
            )

    def _compact_accumulate_observed_step(
        self,
        curr_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
        parent_messages: torch.Tensor,
        child_messages: torch.Tensor,
        pf: torch.Tensor,
        transition_counts: torch.Tensor,
    ) -> torch.Tensor:
        pf_next = torch.zeros_like(pf)
        src_sizes = self.clones_per_token[curr_tokens]
        dst_sizes = self.clones_per_token[next_tokens]
        block_idx, matched = self._observed_pair_block_indices(curr_tokens, next_tokens)

        for src_size, dst_size, cols in self._compact_size_groups(src_sizes, dst_sizes):
            blocks = self._compact_effective_blocks(
                curr_tokens=curr_tokens,
                cols=cols,
                block_idx=block_idx,
                matched=matched,
                src_size=src_size,
                dst_size=dst_size,
            )
            parent = parent_messages[cols, :src_size]
            child = child_messages[cols, :dst_size]
            local_pf = pf[cols, :src_size]
            local_counts = (
                local_pf.unsqueeze(-1)
                * blocks
                * torch.exp(child.unsqueeze(1) - parent.unsqueeze(-1))
            )
            local_counts = torch.nan_to_num(local_counts, nan=0.0, posinf=0.0, neginf=0.0)

            local_matched = matched[cols]
            if local_matched.any():
                local_rows = local_matched.nonzero(as_tuple=False).squeeze(-1)
                local_block_idx = block_idx[cols[local_rows]]
                starts = self.transitions.block_value_start[local_block_idx]
                offsets = starts[:, None] + torch.arange(src_size * dst_size, device=self.device)
                transition_counts.index_add_(
                    0,
                    offsets.reshape(-1),
                    local_counts[local_rows].reshape(-1),
                )

            pf_next[cols, :dst_size] = local_counts.sum(dim=1)

        return pf_next

    def accumulate_observed(
        self,
        input_ids: torch.Tensor,
        transition_counts: torch.Tensor,
        gamma_flow: torch.Tensor,
    ) -> torch.Tensor:
        messages, loglik = self.forward_observed_compact(input_ids)
        self.backward_observed_compact(
            input_ids=input_ids,
            messages=messages,
            loglik=loglik,
            transition_counts=transition_counts,
            gamma_flow=gamma_flow,
        )
        return loglik

    def _compute_local_backward_stats(
        self,
        pf_block: torch.Tensor,
        parent_block: torch.Tensor,
        child_block: torch.Tensor,
        transition_block: torch.Tensor,
        flow_transition_block: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if flow_transition_block is None:
            flow_transition_block = transition_block

        expected_counts = ib_ib_bj_to_ij(
            torch.permute(pf_block, (1, 0)).contiguous(),
            torch.permute(parent_block, (1, 0)).contiguous(),
            child_block,
        )
        expected_counts = expected_counts * transition_block

        parent_max = torch.amax(parent_block, dim=1, keepdim=True)
        parent_max = torch.where(torch.isfinite(parent_max), parent_max, torch.zeros_like(parent_max))
        parent_probs = torch.exp(parent_block - parent_max)
        parent_probs = torch.nan_to_num(parent_probs, nan=0.0, posinf=0.0, neginf=0.0)

        ratio = pf_block / parent_probs
        ratio[parent_probs == 0.0] = 0.0
        ratio = torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

        next_flow = matmul(ratio, flow_transition_block) * torch.exp(child_block - parent_max)
        next_flow = torch.nan_to_num(next_flow, nan=0.0, posinf=0.0, neginf=0.0)
        return expected_counts, next_flow

    def _compute_floor_next_flow(
        self,
        pf_block: torch.Tensor,
        parent_block: torch.Tensor,
        child_block: torch.Tensor,
        source_slice: slice,
    ) -> torch.Tensor:
        parent_max = torch.amax(parent_block, dim=1, keepdim=True)
        parent_max = torch.where(torch.isfinite(parent_max), parent_max, torch.zeros_like(parent_max))
        parent_probs = torch.exp(parent_block - parent_max)
        parent_probs = torch.nan_to_num(parent_probs, nan=0.0, posinf=0.0, neginf=0.0)

        ratio = pf_block / parent_probs
        ratio[parent_probs == 0.0] = 0.0
        ratio = torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

        floor = self.transition_floor[source_slice].unsqueeze(0)
        weighted_floor = torch.sum(ratio * floor, dim=1, keepdim=True)
        next_flow = weighted_floor * torch.exp(child_block - parent_max)
        return torch.nan_to_num(next_flow, nan=0.0, posinf=0.0, neginf=0.0)

    def _iter_token_groups(self, tokens: torch.Tensor, mask: torch.Tensor) -> Iterable[tuple[int, torch.Tensor]]:
        if not mask.any():
            return []
        cols_all = mask.nonzero(as_tuple=False).squeeze(-1)
        token_values = tokens[cols_all]
        groups = []
        for token_id in torch.unique(token_values):
            groups.append((int(token_id.item()), cols_all[token_values == token_id]))
        return groups

    def _iter_pair_groups(
        self,
        curr_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> Iterable[tuple[int, torch.Tensor]]:
        if not mask.any():
            return []
        cols_all = mask.nonzero(as_tuple=False).squeeze(-1)
        pair_codes = curr_tokens[cols_all] * self.vocab_size + next_tokens[cols_all]
        groups = []
        for pair_code in torch.unique(pair_codes):
            groups.append((int(pair_code.item()), cols_all[pair_codes == pair_code]))
        return groups

    def _initialize_last_messages(self, last_tokens: torch.Tensor, batch_size: int) -> torch.Tensor:
        messages = torch.full(
            (self.hidden_states, batch_size),
            float("-inf"),
            device=self.device,
            dtype=self.gamma.dtype,
        )
        observed = last_tokens != -1
        for token_id, cols in self._iter_token_groups(last_tokens, observed):
            messages[self._token_slice(token_id), cols] = 0.0
        if (~observed).any():
            messages[:, ~observed] = 0.0
        return messages

    def _propagate_observed_pairs(
        self,
        curr_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
        next_messages: torch.Tensor,
        new_messages: torch.Tensor,
    ) -> None:
        mask = (curr_tokens != -1) & (next_tokens != -1)
        for pair_code, cols in self._iter_pair_groups(curr_tokens, next_tokens, mask):
            block_idx = self.transitions.block_index_for_pair_code(pair_code)
            if block_idx is None:
                src_token = pair_code // self.vocab_size
                dst_token = pair_code % self.vocab_size
                source_slice = self._token_slice(src_token)
                dest_slice = self._token_slice(dst_token)
                block = self._floor_block(source_slice, dest_slice)
            else:
                spec = self.transitions.block_specs[block_idx]
                source_slice = slice(spec.src_start, spec.src_stop)
                dest_slice = slice(spec.dst_start, spec.dst_stop)
                block = self._effective_block(block_idx)

            child_messages = next_messages[dest_slice, cols]
            new_messages[source_slice, cols] = self._stable_project(block, child_messages)

    def _propagate_observed_to_missing(
        self,
        curr_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
        next_messages: torch.Tensor,
        new_messages: torch.Tensor,
    ) -> None:
        mask = (curr_tokens != -1) & (next_tokens == -1)
        for token_id, cols in self._iter_token_groups(curr_tokens, mask):
            source_slice = self._token_slice(token_id)
            floor_contrib = self._floor_project(source_slice, next_messages[:, cols])
            new_messages[source_slice, cols] = torch.logaddexp(
                new_messages[source_slice, cols],
                floor_contrib,
            )
            for block_idx in self.transitions.outgoing_block_ids[token_id]:
                spec = self.transitions.block_specs[block_idx]
                block = self.transitions.block(block_idx)
                child_messages = next_messages[spec.dst_start : spec.dst_stop, cols]
                contrib = self._stable_project(block, child_messages)
                new_messages[source_slice, cols] = torch.logaddexp(new_messages[source_slice, cols], contrib)

    def _propagate_missing_to_observed(
        self,
        curr_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
        next_messages: torch.Tensor,
        new_messages: torch.Tensor,
    ) -> None:
        mask = (curr_tokens == -1) & (next_tokens != -1)
        for token_id, cols in self._iter_token_groups(next_tokens, mask):
            dest_slice = self._token_slice(token_id)
            child_messages = next_messages[dest_slice, cols]
            all_sources = slice(0, self.hidden_states)
            floor_contrib = self._floor_project(all_sources, child_messages)
            new_messages[:, cols] = torch.logaddexp(new_messages[:, cols], floor_contrib)
            for block_idx in self.transitions.incoming_block_ids[token_id]:
                spec = self.transitions.block_specs[block_idx]
                block = self.transitions.block(block_idx)
                contrib = self._stable_project(block, child_messages)
                source_slice = slice(spec.src_start, spec.src_stop)
                new_messages[source_slice, cols] = torch.logaddexp(new_messages[source_slice, cols], contrib)

    def _propagate_missing_to_missing(
        self,
        curr_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
        next_messages: torch.Tensor,
        new_messages: torch.Tensor,
    ) -> None:
        mask = (curr_tokens == -1) & (next_tokens == -1)
        if not mask.any():
            return
        cols = mask.nonzero(as_tuple=False).squeeze(-1)
        all_sources = slice(0, self.hidden_states)
        floor_contrib = self._floor_project(all_sources, next_messages[:, cols])
        new_messages[:, cols] = torch.logaddexp(new_messages[:, cols], floor_contrib)
        for block_idx, spec in enumerate(self.transitions.block_specs):
            block = self.transitions.block(block_idx)
            child_messages = next_messages[spec.dst_start : spec.dst_stop, cols]
            contrib = self._stable_project(block, child_messages)
            source_slice = slice(spec.src_start, spec.src_stop)
            new_messages[source_slice, cols] = torch.logaddexp(new_messages[source_slice, cols], contrib)

    def forward(self, input_ids: torch.Tensor) -> list[torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        input_ids_t = torch.permute(input_ids, (1, 0)).contiguous()

        ys: list[torch.Tensor] = []
        y = self._initialize_last_messages(input_ids_t[-1], batch_size)
        ys.append(y)

        for t in range(seq_len - 2, -1, -1):
            curr_tokens = input_ids_t[t]
            next_tokens = input_ids_t[t + 1]
            new_y = torch.full_like(y, float("-inf"))

            self._propagate_observed_pairs(curr_tokens, next_tokens, y, new_y)
            self._propagate_observed_to_missing(curr_tokens, next_tokens, y, new_y)
            self._propagate_missing_to_observed(curr_tokens, next_tokens, y, new_y)
            self._propagate_missing_to_missing(curr_tokens, next_tokens, y, new_y)

            y = new_y
            ys.append(y)

        gamma_exp = torch.softmax(self.gamma, dim=0)
        y_max = torch.amax(y, dim=0)
        y_max = torch.where(torch.isfinite(y_max), y_max, torch.zeros_like(y_max))
        y_probs = torch.exp(y - y_max.unsqueeze(0))
        y_probs = torch.nan_to_num(y_probs, nan=0.0, posinf=0.0, neginf=0.0)
        loglik = matmul(gamma_exp.unsqueeze(0), y_probs).squeeze()
        loglik = torch.log(loglik) + y_max

        ys.append(loglik)
        return ys

    def _accumulate_observed_pair_counts(
        self,
        curr_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
        parent_messages: torch.Tensor,
        child_messages: torch.Tensor,
        pf: torch.Tensor,
        transition_counts: torch.Tensor,
        pf_next: torch.Tensor,
    ) -> None:
        mask = (curr_tokens != -1) & (next_tokens != -1)
        for pair_code, cols in self._iter_pair_groups(curr_tokens, next_tokens, mask):
            block_idx = self.transitions.block_index_for_pair_code(pair_code)
            if block_idx is None:
                src_token = pair_code // self.vocab_size
                dst_token = pair_code % self.vocab_size
                source_slice = self._token_slice(src_token)
                dest_slice = self._token_slice(dst_token)
                block = self._floor_block(source_slice, dest_slice)
                spec = None
            else:
                spec = self.transitions.block_specs[block_idx]
                block = self._effective_block(block_idx)
                source_slice = slice(spec.src_start, spec.src_stop)
                dest_slice = slice(spec.dst_start, spec.dst_stop)

            pf_block = pf[cols, source_slice]
            parent_block = torch.permute(parent_messages[source_slice, cols], (1, 0))
            child_block = torch.permute(child_messages[dest_slice, cols], (1, 0))

            local_counts, local_next_flow = self._compute_local_backward_stats(
                pf_block=pf_block,
                parent_block=parent_block,
                child_block=child_block,
                transition_block=block,
            )
            if spec is not None:
                self.transitions.block_view(transition_counts, spec).add_(local_counts)
            pf_next[cols, dest_slice] = local_next_flow

    def _accumulate_observed_to_missing_counts(
        self,
        curr_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
        parent_messages: torch.Tensor,
        child_messages: torch.Tensor,
        pf: torch.Tensor,
        transition_counts: torch.Tensor,
        pf_next: torch.Tensor,
    ) -> None:
        mask = (curr_tokens != -1) & (next_tokens == -1)
        for token_id, cols in self._iter_token_groups(curr_tokens, mask):
            source_slice = self._token_slice(token_id)
            pf_block = pf[cols, source_slice]
            parent_block = torch.permute(parent_messages[source_slice, cols], (1, 0))
            all_child_block = torch.permute(child_messages[:, cols], (1, 0))
            pf_next[cols, :] += self._compute_floor_next_flow(
                pf_block=pf_block,
                parent_block=parent_block,
                child_block=all_child_block,
                source_slice=source_slice,
            )

            for block_idx in self.transitions.outgoing_block_ids[token_id]:
                spec = self.transitions.block_specs[block_idx]
                count_block = self._effective_block(block_idx)
                flow_block = self.transitions.block(block_idx)
                dest_slice = slice(spec.dst_start, spec.dst_stop)
                child_block = torch.permute(child_messages[dest_slice, cols], (1, 0))

                local_counts, local_next_flow = self._compute_local_backward_stats(
                    pf_block=pf_block,
                    parent_block=parent_block,
                    child_block=child_block,
                    transition_block=count_block,
                    flow_transition_block=flow_block,
                )
                self.transitions.block_view(transition_counts, spec).add_(local_counts)
                pf_next[cols, dest_slice] += local_next_flow

    def _accumulate_missing_to_observed_counts(
        self,
        curr_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
        parent_messages: torch.Tensor,
        child_messages: torch.Tensor,
        pf: torch.Tensor,
        transition_counts: torch.Tensor,
        pf_next: torch.Tensor,
    ) -> None:
        mask = (curr_tokens == -1) & (next_tokens != -1)
        for token_id, cols in self._iter_token_groups(next_tokens, mask):
            all_sources = slice(0, self.hidden_states)
            dest_slice = self._token_slice(token_id)
            child_block = torch.permute(child_messages[dest_slice, cols], (1, 0))
            all_pf_block = pf[cols, :]
            all_parent_block = torch.permute(parent_messages[:, cols], (1, 0))
            pf_next[cols, dest_slice] += self._compute_floor_next_flow(
                pf_block=all_pf_block,
                parent_block=all_parent_block,
                child_block=child_block,
                source_slice=all_sources,
            )

            for block_idx in self.transitions.incoming_block_ids[token_id]:
                spec = self.transitions.block_specs[block_idx]
                count_block = self._effective_block(block_idx)
                flow_block = self.transitions.block(block_idx)
                source_slice = slice(spec.src_start, spec.src_stop)
                pf_block = pf[cols, source_slice]
                parent_block = torch.permute(parent_messages[source_slice, cols], (1, 0))

                local_counts, local_next_flow = self._compute_local_backward_stats(
                    pf_block=pf_block,
                    parent_block=parent_block,
                    child_block=child_block,
                    transition_block=count_block,
                    flow_transition_block=flow_block,
                )
                self.transitions.block_view(transition_counts, spec).add_(local_counts)
                pf_next[cols, dest_slice] += local_next_flow

    def _accumulate_missing_to_missing_counts(
        self,
        curr_tokens: torch.Tensor,
        next_tokens: torch.Tensor,
        parent_messages: torch.Tensor,
        child_messages: torch.Tensor,
        pf: torch.Tensor,
        transition_counts: torch.Tensor,
        pf_next: torch.Tensor,
    ) -> None:
        mask = (curr_tokens == -1) & (next_tokens == -1)
        if not mask.any():
            return
        cols = mask.nonzero(as_tuple=False).squeeze(-1)
        all_sources = slice(0, self.hidden_states)
        all_pf_block = pf[cols, :]
        all_parent_block = torch.permute(parent_messages[:, cols], (1, 0))
        all_child_block = torch.permute(child_messages[:, cols], (1, 0))
        pf_next[cols, :] += self._compute_floor_next_flow(
            pf_block=all_pf_block,
            parent_block=all_parent_block,
            child_block=all_child_block,
            source_slice=all_sources,
        )

        for block_idx, spec in enumerate(self.transitions.block_specs):
            count_block = self._effective_block(block_idx)
            flow_block = self.transitions.block(block_idx)
            source_slice = slice(spec.src_start, spec.src_stop)
            dest_slice = slice(spec.dst_start, spec.dst_stop)
            pf_block = pf[cols, source_slice]
            parent_block = torch.permute(parent_messages[source_slice, cols], (1, 0))
            child_block = torch.permute(child_messages[dest_slice, cols], (1, 0))

            local_counts, local_next_flow = self._compute_local_backward_stats(
                pf_block=pf_block,
                parent_block=parent_block,
                child_block=child_block,
                transition_block=count_block,
                flow_transition_block=flow_block,
            )
            self.transitions.block_view(transition_counts, spec).add_(local_counts)
            pf_next[cols, dest_slice] += local_next_flow

    def backward(
        self,
        input_ids: torch.Tensor,
        probs: list[torch.Tensor],
        transition_counts: torch.Tensor,
        beta_flow: torch.Tensor | None,
        gamma_flow: torch.Tensor,
    ) -> None:
        del beta_flow

        batch_size, seq_len = input_ids.shape
        input_ids_t = torch.permute(input_ids, (1, 0)).contiguous()
        gamma_exp = torch.softmax(self.gamma, dim=0)

        pf = gamma_exp.unsqueeze(0) * torch.exp(
            torch.permute(probs[-2], (1, 0)).contiguous() - probs[-1][:, None]
        )
        pf = torch.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)
        gamma_flow.add_(torch.sum(pf, dim=0))

        for t in range(seq_len - 1):
            layer_idx = seq_len - t - 1
            parent_messages = probs[layer_idx]
            child_messages = probs[layer_idx - 1]
            curr_tokens = input_ids_t[t]
            next_tokens = input_ids_t[t + 1]
            pf_next = torch.zeros_like(pf)

            self._accumulate_observed_pair_counts(
                curr_tokens,
                next_tokens,
                parent_messages,
                child_messages,
                pf,
                transition_counts,
                pf_next,
            )
            self._accumulate_observed_to_missing_counts(
                curr_tokens,
                next_tokens,
                parent_messages,
                child_messages,
                pf,
                transition_counts,
                pf_next,
            )
            self._accumulate_missing_to_observed_counts(
                curr_tokens,
                next_tokens,
                parent_messages,
                child_messages,
                pf,
                transition_counts,
                pf_next,
            )
            self._accumulate_missing_to_missing_counts(
                curr_tokens,
                next_tokens,
                parent_messages,
                child_messages,
                pf,
                transition_counts,
                pf_next,
            )

            pf = torch.nan_to_num(pf_next, nan=0.0, posinf=0.0, neginf=0.0)

    def loglikelihood(self, input_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
        ll = torch.tensor([0.0], device=self.device, dtype=self.gamma.dtype)
        for batch_idx in range(0, input_ids.shape[0], batch_size):
            input_ids_batch_cpu = input_ids[batch_idx : batch_idx + batch_size]
            has_missing = torch.any(input_ids_batch_cpu == -1)
            input_ids_batch = input_ids_batch_cpu.to(self.device)
            if not has_missing:
                _, loglik = self.forward_observed_compact(input_ids_batch)
                ll += torch.sum(loglik)
            else:
                probs = self.forward(input_ids_batch)
                ll += torch.sum(probs[-1])
        return ll
