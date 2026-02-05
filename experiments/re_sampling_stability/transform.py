"""Problem transformation functions for re-sampling stability experiments.

The main entry point used by ``run_experiment_async`` is a pluggable async
function with the following informal signature::

    async def transform_problems(config, problem) -> list[problem]:

In this module we provide a concrete implementation,
``permutate_actions(config, problem)``, that can be referenced from
experiment configuration (YAML or CLI) as a dotted path, e.g.::

    experiments.re_sampling_stability.transform.permutate_actions

The behavior implemented here is intentionally simple and conservative:

- Always include the base problem unchanged.
- Optionally add several variants where the *order* of all actions is
  randomly permuted.

The goal of this transformation is to study how stable model judgments
are under re-sampling of option order. We therefore avoid attaching any
additional metadata here and keep the return type as plain
``PracticalProblem`` objects. For transformed variants we construct fresh
``PracticalProblem`` instances from the decision situation and a
permuted ``actions`` list, so any extra fields present on the original
problem are not preserved.
"""

from __future__ import annotations

import itertools
import logging
import random
from typing import Any, List

from practical_deliberation_llms.model import PracticalProblem


logger = logging.getLogger(__name__)


async def permutate_actions(
    config: Any, problem: PracticalProblem
) -> List[PracticalProblem]:
    """Return a list of problems with randomly permuted action orders.

    The contract for transform functions in this experiment code is:

    - Always include the *original* problem as the first element of the
      returned list.
    - Honor ``config.max_transformations_per_problem`` as an upper bound on
      the number of additional variants per base problem.
    - Do not mutate the input ``problem`` in-place; always create new
      ``PracticalProblem`` instances for transformed variants.

    This implementation treats re-sampling purely as *reordering* of the
    available actions. For a problem with ``n`` actions we:

    - Enumerate all permutations of the indices ``0 .. n-1``.
    - Exclude the identity permutation (which would duplicate the
      unchanged base problem that we already include).
    - Shuffle the remaining permutations using Python's RNG.
    - Take up to ``max_transformations_per_problem`` permutations and build
      new ``PracticalProblem`` instances with the corresponding permuted
      ``actions`` lists.

    This guarantees that, within a single call, no two transformed
    variants share the same action order, and none of them is identical to
    the original problem. We intentionally do *not* attach any extra
    metadata here; the focus is on the option-order manipulation itself.
    """

    max_transforms = getattr(config, "max_transformations_per_problem", 0)

    logger.debug(
        "permutate_actions start problem_uid=%s max_transforms=%d",
        getattr(problem, "problem_uid", None),
        max_transforms,
    )

    # Always include the original problem as the first element. Downstream
    # code (metrics, plotting) relies on this convention.
    results: List[PracticalProblem] = [problem]

    # Honor the config knob that allows callers to completely disable
    # transformed variants while still wiring in this transform function.
    if max_transforms <= 0:
        logger.debug("permutate_actions disabled_by_config returning_base_only")
        return results

    n_actions = len(problem.actions)

    # If there are fewer than two actions, there is nothing to permute.
    if n_actions < 2:
        logger.debug(
            "permutate_actions skip_insufficient_actions problem_uid=%s n_actions=%d",
            getattr(problem, "problem_uid", None),
            n_actions,
        )
        return results

    # We work with permutations of indices rather than actions themselves so
    # that the logic is independent of how actions are represented.
    indices = list(range(n_actions))
    identity = tuple(indices)

    all_perms = [perm for perm in itertools.permutations(indices) if perm != identity]

    # Randomize the order in which we consider permutations. The global
    # RNG seed is controlled by ``set_global_seeds`` in run_experiment.py,
    # so this remains reproducible across runs given a fixed experiment
    # seed.
    random.shuffle(all_perms)

    # Respect the upper bound from the config, but never try to create more
    # variants than there are distinct non-identity permutations.
    selected_perms = all_perms[:max_transforms]

    for perm in selected_perms:
        permuted_actions = [problem.actions[i] for i in perm]

        # Construct a fresh PracticalProblem with permuted actions. We rely
        # on the dataclass to regenerate labels in a way that is consistent
        # with the new action order.
        permuted_problem = PracticalProblem(
            decision_situation=problem.decision_situation,
            actions=permuted_actions,
        )

        results.append(permuted_problem)

    logger.debug(
        "permutate_actions created_variants problem_uid=%s n_variants=%d",
        getattr(problem, "problem_uid", None),
        len(results) - 1,
    )

    return results
