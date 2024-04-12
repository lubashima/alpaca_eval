from typing import Sequence, Union

import pandas as pd

from alpaca_eval import utils

from .helpers import AbsoluteScoringRule, ZeroOneScoringRule

__all__ = ["get_winrate", "pairwise_to_winrate"]

def is_longer(generation_one, generation_two):
    if len(generation_one) > len(generation_two):
        return 1.0 
    else:
        return 2.0
    
def get_first_order_bias(annotations):
    return annotations.raw_completion.str.contains('Output \(a\)').sum()

def get_winrate(annotations: Union[pd.DataFrame, Sequence]) -> dict[str, float]:
    """Extract head2head metrics (n_wins, n_counts, win_rate) from a sequence preference.
    This assumes that the preference is encoded as 0 or 1.5 for draw, 1 for base win, 2 when the model to compare wins.
    """
    annotations = utils.convert_to_dataframe(annotations)
    preferences = annotations["preference"]
    print(annotations)
    annotations['compare_length'] = [is_longer(a, b) for a, b in zip(annotations['output_1'], annotations['output_2'])]
    out = AbsoluteScoringRule().describe_head2head(preferences)
    print(f"Matches: {get_first_order_bias(annotations)}")
    out['first_order_preference'] = get_first_order_bias(annotations) / out['n_total']
    out["reference"] = annotations['generator_1'].unique()[0]
    out["generator"] = annotations['generator_2'].unique()[0]
    out["discrete_win_rate"] = ZeroOneScoringRule().describe_head2head(preferences)["win_rate"]
    out["length_bias"] = sum(annotations['compare_length'] == annotations['preference']) / out['n_total']
    out["ego_bias"] = out['n_wins_base'] / out['n_total']
    print(out)
    return out


# backward compatibility
def pairwise_to_winrate(preferences: Union[pd.DataFrame, Sequence]) -> dict[str, float]:
    """Extract head2head metrics (n_wins, n_counts, win_rate) from a sequence preference.
    This assumes that the preference is encoded as 0 or 1.5 for draw, 1 for base win, 2 when the model to compare wins.
    """
    return get_winrate(annotations=[dict(preferences=p) for p in preferences])
