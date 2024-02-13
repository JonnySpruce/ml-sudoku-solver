from collections import Counter


def check_duplicates(row: list[int]) -> bool:
    counts = Counter()
    for cell in row:
        if cell != 0:
            counts[cell] += 1
        if cell > 9 or counts[cell] > 1:
            return False
    return True


def check_sudoku(grid: list[list[int]]) -> bool:
    if len(grid) != 9 or (sum(len(row) == 9 for row in grid) != 9):
        return False
    for row in grid:
        if not check_duplicates(row):
            return False
    return True
