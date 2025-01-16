from typing import Dict, Any
from optuna.distributions import BaseDistribution , FloatDistribution , IntDistribution , \
    CategoricalDistribution


def continuity_and_smoothness(search_space: Dict[str , BaseDistribution]) -> Dict[
    str , Dict[str , str]] :
    """
    Analyze continuity and smoothness of distributions in the search space.

    Args:
        search_space (Dict[str, BaseDistribution]): A dictionary of parameter names and their distributions.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary with parameter names as keys and their continuity and smoothness as values.
    """

    def assess_continuity(distribution: BaseDistribution) -> str :
        """Determine continuity based on distribution type."""
        if isinstance(distribution , FloatDistribution) :
            return "Continuous"
        elif isinstance(distribution , IntDistribution) or isinstance(distribution ,
                                                                      CategoricalDistribution) :
            return "Discrete"
        else :
            return "Unknown"

    def assess_smoothness(distribution: BaseDistribution) -> str :
        """
        Determine smoothness based on distribution type or additional heuristics.

        Smoothness can also depend on specific problem details, which can be
        provided as user-defined rules or estimated using sampling.
        """
        if isinstance(distribution , FloatDistribution) :
            return "Smooth"  # Assumes float distributions map to smooth functions
        elif isinstance(distribution , IntDistribution) :
            return "Smooth"  # Assumes nearby integers produce similar outputs
        elif isinstance(distribution , CategoricalDistribution) :
            return "Not Smooth"  # Assumes categorical variables have no inherent smoothness
        else :
            return "Unknown"

    # Main analysis
    analysis = { }
    for param_name , distribution in search_space.items() :
        continuity = assess_continuity(distribution)
        smoothness = assess_smoothness(distribution)
        analysis[param_name] = {
            "Continuity" : continuity ,
            "Smoothness" : smoothness
            }

    return analysis


def size(search_space: Dict[str, BaseDistribution]) -> Dict[str, Any]:
    """
    Analyze the dimensionality and size of the search space.

    Args:
        search_space (Dict[str, BaseDistribution]): A dictionary of parameter names and their distributions.

    Returns:
        Dict[str, Any]: A dictionary containing the total number of dimensions,
        number of dimensions with infinite available values, number of dimensions with finite values,
        and the total size of the search space for finite dimensions.
    """
    infinite_dimensions = 0
    finite_dimensions = 0
    finite_sizes = []

    for param_name, distribution in search_space.items():
        if isinstance(distribution, FloatDistribution):
            # Float distributions have infinite available values.
            infinite_dimensions += 1
        elif isinstance(distribution, IntDistribution):
            # Int distributions have a finite range of integer values.
            finite_dimensions += 1
            size = distribution.high - distribution.low + 1
            finite_sizes.append(size)
        elif isinstance(distribution, CategoricalDistribution):
            # Categorical distributions have a finite set of values.
            finite_dimensions += 1
            size = len(distribution.choices)
            finite_sizes.append(size)
        else:
            # Unknown distribution types are ignored for now.
            pass

    total_finite_size = 1
    for size in finite_sizes:
        total_finite_size *= size

    if finite_dimensions == 0 :
        total_finite_size = 0

    return {
        "Total Dimensions": len(search_space),
        "Infinite Dimensions": infinite_dimensions,
        "Finite Dimensions": finite_dimensions,
        "Total Finite Size": total_finite_size
    }

