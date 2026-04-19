from pathlib import Path

import pytest

from headkv.config import validate_headkv_matrix_csv_shape


def write_csv(path: Path, rows: list[list[object]]) -> None:
    path.write_text(
        "\n".join(",".join(str(cell) for cell in row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_validate_headkv_matrix_csv_shape_accepts_matching_matrix(tmp_path: Path) -> None:
    csv_path = tmp_path / "classification.csv"
    write_csv(csv_path, [[-1, 1, 2], [1, 1, -1]])

    validate_headkv_matrix_csv_shape(str(csv_path), num_layers=2, num_heads=3)


def test_validate_headkv_matrix_csv_shape_ignores_triplet_capacity_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "capacity.csv"
    write_csv(csv_path, [[0, 0, 1024], [1, 2, 2048]])

    validate_headkv_matrix_csv_shape(str(csv_path), num_layers=30, num_heads=12)


def test_validate_headkv_matrix_csv_shape_rejects_mismatched_matrix(tmp_path: Path) -> None:
    csv_path = tmp_path / "classification.csv"
    write_csv(csv_path, [[-1, 1], [1, 1], [2, -1]])

    with pytest.raises(ValueError, match="shape mismatch"):
        validate_headkv_matrix_csv_shape(str(csv_path), num_layers=2, num_heads=3)
