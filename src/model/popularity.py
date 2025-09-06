
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import pdb


class PopularityEncoding(torch.nn.Module):
    """Popularity-based encoding for sequential models."""

    def __init__(
        self,
        input_units1: int,
        input_units2: int,
        base_dim1: int,
        base_dim2: int,
        popularity_dir,
        dataset: str,
    ):
        super().__init__()
        self.input1 = input_units1
        self.input2 = input_units2
        self.base_dim1 = base_dim1
        self.base_dim2 = base_dim2

        pop_dir = Path(popularity_dir)
        month_pop = np.loadtxt(pop_dir / f"{dataset}_month_pop.txt")
        week_pop = np.loadtxt(pop_dir / f"{dataset}_week_pop.txt")

        self.register_buffer(
            "month_pop_table",
            torch.cat(
                (
                    torch.zeros((month_pop.shape[0] + self.input1 - self.base_dim1, 1)),
                    torch.cat(
                        (
                            torch.zeros(
                                (self.input1 - self.base_dim1, month_pop.shape[1])
                            ),
                            torch.FloatTensor(month_pop),
                        ),
                        dim=0,
                    ),
                ),
                dim=1,
            ),
        )
        self.register_buffer(
            "week_pop_table",
            torch.cat(
                (
                    torch.zeros((week_pop.shape[0] + self.input2 - self.base_dim2, 1)),
                    torch.cat(
                        (
                            torch.zeros(
                                (self.input2 - self.base_dim2, week_pop.shape[1])
                            ),
                            torch.FloatTensor(week_pop),
                        ),
                        dim=0,
                    ),
                ),
                dim=1,
            ),
        )

    def forward(self, log_seqs, time1_seqs, time2_seqs):
        month_table_rows = torch.flatten(
            torch.flatten(torch.LongTensor(time1_seqs)).reshape((-1, 1))
            * self.base_dim1
            + torch.arange(self.input1)
        )
        month_table_cols = torch.repeat_interleave(
            torch.flatten(torch.LongTensor(log_seqs)), self.input1
        )
        week_table_rows = torch.flatten(
            torch.flatten(torch.LongTensor(time2_seqs)).reshape((-1, 1))
            * self.base_dim2
            + torch.arange(self.input2)
        )
        week_table_cols = torch.repeat_interleave(
            torch.flatten(torch.LongTensor(log_seqs)), self.input2
        )
        if (
            (month_table_rows.numel() > 0 and torch.max(month_table_rows) >= self.month_pop_table.shape[0])
            or (month_table_cols.numel() > 0 and torch.max(month_table_cols) >= self.month_pop_table.shape[1])
            or (week_table_rows.numel() > 0 and torch.max(week_table_rows) >= self.week_pop_table.shape[0])
            or (week_table_cols.numel() > 0 and torch.max(week_table_cols) >= self.week_pop_table.shape[1])
        ):
            pdb.set_trace()
        month_pop = torch.reshape(
            self.month_pop_table[month_table_rows, month_table_cols],
            (log_seqs.shape[0], log_seqs.shape[1], self.input1),
        )
        week_pop = torch.reshape(
            self.week_pop_table[week_table_rows, week_table_cols],
            (log_seqs.shape[0], log_seqs.shape[1], self.input2),
        )
        return torch.cat((month_pop, week_pop), 2).clone().detach()


class EvalPopularityEncoding(torch.nn.Module):
    """Popularity encoding used during evaluation with recent statistics."""

    def __init__(
        self,
        input_units1: int,
        input_units2: int,
        base_dim1: int,
        base_dim2: int,
        popularity_dir,
        dataset: str,
        pause: bool = False,
    ):
        super().__init__()
        self.input1 = input_units1
        self.input2 = input_units2
        self.base_dim1 = base_dim1
        self.base_dim2 = base_dim2
        self.pause = pause

        pop_dir = Path(popularity_dir)
        month_pop = np.loadtxt(pop_dir / f"{dataset}_month_pop.txt")
        week_pop = np.loadtxt(pop_dir / f"{dataset}_week_pop.txt")
        week_eval_pop = np.loadtxt(pop_dir / f"{dataset}_week_eval_pop.txt")

        self.register_buffer("week_eval_pop", torch.FloatTensor(week_eval_pop))
        self.register_buffer(
            "month_pop_table",
            torch.cat(
                (
                    torch.zeros((month_pop.shape[0] + self.input1 - self.base_dim1, 1)),
                    torch.cat(
                        (
                            torch.zeros(
                                (self.input1 - self.base_dim1, month_pop.shape[1])
                            ),
                            torch.FloatTensor(month_pop),
                        ),
                        dim=0,
                    ),
                ),
                dim=1,
            ),
        )
        self.register_buffer(
            "week_pop_table",
            torch.cat(
                (
                    torch.zeros((week_pop.shape[0] + self.input2 - self.base_dim2, 1)),
                    torch.cat(
                        (
                            torch.zeros(
                                (self.input2 - self.base_dim2, week_pop.shape[1])
                            ),
                            torch.FloatTensor(week_pop),
                        ),
                        dim=0,
                    ),
                ),
                dim=1,
            ),
        )

    def forward(self, log_seqs, time1_seqs, time2_seqs, user):
        month_table_rows = torch.flatten(
            torch.flatten(torch.LongTensor(time1_seqs)).reshape((-1, 1))
            * self.base_dim1
            + torch.arange(self.input1)
        )
        month_table_cols = torch.repeat_interleave(
            torch.flatten(torch.LongTensor(log_seqs)), self.input1
        )
        if self.input2 > self.base_dim2:
            week_table_rows = torch.flatten(
                torch.flatten(torch.LongTensor(time2_seqs)).reshape((-1, 1))
                * self.base_dim2
                + torch.arange(self.input2 - self.base_dim2)
            )
            week_table_cols = torch.repeat_interleave(
                torch.flatten(torch.LongTensor(log_seqs)), self.input2 - self.base_dim2
            )
            if (
                torch.max(week_table_rows) >= self.week_pop_table.shape[0]
                or torch.max(week_table_cols) >= self.week_pop_table.shape[1]
            ):
                raise IndexError("row or column accessed out-of-index in popularity table")

        if (
            torch.max(month_table_rows) >= self.month_pop_table.shape[0]
            or torch.max(month_table_cols) >= self.month_pop_table.shape[1]
        ):
            raise IndexError("row or column accessed out-of-index in popularity table")
        month_pop = torch.reshape(
            self.month_pop_table[month_table_rows, month_table_cols],
            (log_seqs.shape[0], log_seqs.shape[1], self.input1),
        )

        week_eval_rows = torch.flatten(
            (torch.LongTensor(user - 1) * self.base_dim2).unsqueeze(1)
            + torch.arange(self.base_dim2)
        )
        recent_pop = torch.swapaxes(
            self.week_eval_pop[week_eval_rows].reshape((len(user), 6, -1)), 1, 2
        )
        if self.input2 > self.base_dim2:
            week_pop = torch.reshape(
                self.week_pop_table[week_table_rows, week_table_cols],
                (log_seqs.shape[0], log_seqs.shape[1], self.input2 - self.base_dim2),
            )
            return torch.cat((month_pop, week_pop, recent_pop), 2).clone().detach()
        else:
            return torch.cat((month_pop, recent_pop), 2).clone().detach()


def build_popularity_encoding(
    input_units1: int,
    input_units2: int,
    base_dim1: int,
    base_dim2: int,
    popularity_dir,
    dataset: str,
    enable_eval: bool = False,
    pause: bool = False,
):
    """Factory to create popularity encoders.

    Args:
        input_units1: number of popularity features for month granularity.
        input_units2: number of popularity features for week granularity.
        base_dim1: base dimension for month features.
        base_dim2: base dimension for week features.
        popularity_dir: directory containing popularity files.
        enable_eval: whether to create :class:`EvalPopularityEncoding`.
        pause: extra flag forwarded to :class:`EvalPopularityEncoding`.
    """
    if enable_eval:
        return EvalPopularityEncoding(
            input_units1,
            input_units2,
            base_dim1,
            base_dim2,
            popularity_dir,
            dataset,
            pause=pause,
        )
    return PopularityEncoding(
        input_units1, input_units2, base_dim1, base_dim2, popularity_dir, dataset
    )

