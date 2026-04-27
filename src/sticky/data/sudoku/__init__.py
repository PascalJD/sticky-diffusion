from .boards import (
    board_to_string,
    build_board_batch,
    build_clue_board,
    build_clue_mask,
    build_solution_board,
    boards_to_strings,
    extract_shah_triples,
    get_sudoku_board_num_examples,
    make_sudoku_board_iterator,
    string_to_board,
)
from .download import ensure_sudoku_data_available
from .slack import compute_slack_vectors

__all__ = [
    "ensure_sudoku_data_available",
    "board_to_string",
    "boards_to_strings",
    "string_to_board",
    "extract_shah_triples",
    "build_solution_board",
    "build_clue_board",
    "build_clue_mask",
    "build_board_batch",
    "compute_slack_vectors",
    "get_sudoku_board_num_examples",
    "make_sudoku_board_iterator",
]
