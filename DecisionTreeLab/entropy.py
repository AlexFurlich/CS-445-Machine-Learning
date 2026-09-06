import math
def weighted_entropy(counts, weights):
    """ Weighted entropy across a set of child nodes.

    Args:
        counts: list of class-count lists, one per child node.
        weights: list of weights (n_child / n_parent), one per child.
    """
    return sum(w * entropy(c) for c, w in zip(counts, weights))


def entropy(counts):

    """
    Calculate the entropy of a distribution given the counts of each class.

    Parameters:
    counts (list): A list of counts for each class.

    Returns:
    float: The entropy value.
    """
    total = sum(counts)
    if total == 0:
        return 0.0

    entropy_value = 0.0
    for count in counts:
        if count > 0:
            probability = count / total
            entropy_value += probability * math.log2(probability)
    # if entropy_value == 0:
    #     return True
    return (-1 * entropy_value)

if __name__ == '__main__':
    print(entropy([13,20]))
    print(entropy([1,1,1,1]))
    print(entropy([10,0]))
    print(entropy([4,3]))
    print(weighted_entropy([[3,0], [1,3]], [3/7, 4/7]))
    print("72.5")
    print(weighted_entropy([[0,1], [3,3]], [1/7, 6/7]))
    print("77.5")
    print(weighted_entropy([[1,1], [2,3]], [2/7, 5/7]))
    print("82.5")
    print(weighted_entropy([[2,1], [1,3]], [3/7, 4/7]))
    print("92.5")
    print(weighted_entropy([[2,2], [1,2]], [4/7, 3/7]))
    print("110")
    print(weighted_entropy([[2,3], [1,1]], [5/7, 2/7]))
    print("135")
    print(weighted_entropy([[2,4], [1,0]], [6/7, 1/7]))