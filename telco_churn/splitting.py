from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from telco_churn.config import SplitConfig


@dataclass(frozen=True)
class DatasetSplits:
    """Disjoint train, validation and test partitions of the raw records.

    Attributes:
        train: Records used to fit preprocessing and classifiers.
        validation: Records used to compare candidates and select one.
        test: Records reserved; never used during candidate selection.
        id_column: Customer identifier column shared by the three frames.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    id_column: str

    def items(self) -> list[tuple[str, pd.DataFrame]]:
        """Lists the partitions with their names.

        Returns:
            ``(name, frame)`` pairs in train, validation, test order.
        """
        return [("train", self.train), ("validation", self.validation), ("test", self.test)]

    def allocation(self) -> pd.DataFrame:
        """Records which customer went to which partition.

        Returns:
            One row per customer with the ID and the partition name, sorted by
            ID so the file is stable across runs.
        """
        frames = [
            pd.DataFrame({self.id_column: frame[self.id_column], "split": name})
            for name, frame in self.items()
        ]
        return pd.concat(frames).sort_values(self.id_column).reset_index(drop=True)

    def assert_disjoint(self) -> None:
        """Checks that no customer ID appears in two partitions.

        Raises:
            AssertionError: If any customer ID is shared between partitions.
        """
        ids = {name: set(frame[self.id_column]) for name, frame in self.items()}
        for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            shared = ids[first] & ids[second]
            if shared:
                raise AssertionError(
                    f"{len(shared)} customer ID(s) shared by {first} and {second}, "
                    f"e.g. {sorted(shared)[:3]}."
                )

    def sizes(self) -> dict[str, int]:
        """Counts records per partition.

        Returns:
            Mapping from partition name to number of rows.
        """
        return {name: len(frame) for name, frame in self.items()}


class DatasetSplitter:
    """Reproducible stratified train/validation/test split.

    The split happens in two stratified stages with the same seed: first
    train versus the rest, then the rest into validation and test. The same
    seed and the same input always give the same customer allocation.

    Attributes:
        config: Seed and fractions.
        id_column: Customer identifier column.
        target_column: Column used for stratification.
    """

    def __init__(self, config: SplitConfig, id_column: str, target_column: str) -> None:
        """Initialises the splitter.

        Args:
            config: Seed and fractions.
            id_column: Customer identifier column.
            target_column: Column used for stratification.
        """
        self.config = config
        self.id_column = id_column
        self.target_column = target_column

    def split(self, records: pd.DataFrame) -> DatasetSplits:
        """Splits raw records into disjoint stratified partitions.

        Args:
            records: Raw records with unique customer IDs and a target column.

        Returns:
            The three partitions.

        Raises:
            ValueError: If customer IDs are not unique.
            AssertionError: If the resulting partitions overlap.
        """
        if records[self.id_column].duplicated().any():
            raise ValueError(f"Duplicate {self.id_column} values; cannot split by customer.")
        holdout = self.config.validation_fraction + self.config.test_fraction
        train, rest = train_test_split(
            records,
            test_size=holdout,
            stratify=records[self.target_column],
            random_state=self.config.seed,
        )
        validation, test = train_test_split(
            rest,
            test_size=self.config.test_fraction / holdout,
            stratify=rest[self.target_column],
            random_state=self.config.seed,
        )
        splits = DatasetSplits(
            train=train, validation=validation, test=test, id_column=self.id_column
        )
        splits.assert_disjoint()
        return splits
