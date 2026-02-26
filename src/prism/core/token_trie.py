"""Trie structure for efficiently finding branching points in token sequences."""

from dataclasses import dataclass
from typing import Dict, List

from prism.utils import get_logger

logger = get_logger(__name__)


@dataclass
class BranchPoint:
    """Point where label token sequences diverge.

    At each branch point we compute probabilities over the possible next tokens.
    """

    prefix: List[int]
    branches: Dict[int, List[str]]


class LabelTokenTrie:
    """Build branching structure of label token sequences.

    Example::

        labels = {
            'label1': [1, 3, 7],
            'label2': [2, 4, 3, 6],
            'label3': [2, 4, 3, 8],
            'label4': [2, 5, 1, 7],
        }

        Branch points:
        1. prefix=[], branches={1: ['label1'], 2: ['label2', 'label3', 'label4']}
        2. prefix=[2], branches={4: ['label2', 'label3'], 5: ['label4']}
        3. prefix=[2, 4, 3], branches={6: ['label2'], 8: ['label3']}
    """

    def __init__(self, label_token_sequences: Dict[str, List[int]]):
        self.label_sequences = label_token_sequences
        logger.info(f"Building trie for {len(label_token_sequences)} labels")
        self.branch_points = self._find_branch_points()
        logger.info(f"Found {len(self.branch_points)} branch points")

    def _find_branch_points(self) -> List[BranchPoint]:
        branch_points: List[BranchPoint] = []
        self._find_branches_recursive(
            prefix=[],
            active_labels=list(self.label_sequences.keys()),
            branch_points=branch_points,
        )
        return branch_points

    def _find_branches_recursive(
        self,
        prefix: List[int],
        active_labels: List[str],
        branch_points: List[BranchPoint],
    ):
        if not active_labels:
            return

        next_token_groups: Dict[int, List[str]] = {}

        for label in active_labels:
            seq = self.label_sequences[label]
            if len(seq) == len(prefix):
                continue
            next_token = seq[len(prefix)]
            if next_token not in next_token_groups:
                next_token_groups[next_token] = []
            next_token_groups[next_token].append(label)

        if len(next_token_groups) == 0:
            return
        elif len(next_token_groups) == 1:
            next_token = next(iter(next_token_groups))
            labels = next_token_groups[next_token]
            self._find_branches_recursive(prefix + [next_token], labels, branch_points)
        else:
            branch_points.append(
                BranchPoint(prefix=prefix.copy(), branches=next_token_groups.copy())
            )
            for next_token, labels in next_token_groups.items():
                self._find_branches_recursive(prefix + [next_token], labels, branch_points)
