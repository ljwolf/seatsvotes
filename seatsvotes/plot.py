import matplotlib.pyplot as plt
import numpy as np
from warnings import warn


def cdf_shroud(reference, simulations, ax=None, label=None,
               shroud_kws=None, reference_kws=None, figure_kws=None,
               **common_args):
    if ax is None:
        if figure_kws is None:
            figure_kws = dict()
        f, ax = plt.subplots(1, 1, **figure_kws)
    shroud_kws = dict() if shroud_kws is None else shroud_kws
    reference_kws = dict() if reference_kws is None else reference_kws
    if not isinstance(label, str):
        try:
            reference_label, shroud_label = label
        except ValueError:
            warn('Too many labels passed, taking the first two only')
            reference_label, shroud_label, *rest = label
        except TypeError:
            pass
    if label is not None:
        reference_label = label
        shroud_label = None
    else:
        reference_label = reference_kws.get('label')
        shroud_label = shroud_kws.get('label')
    try:
        del shroud_kws['label']
    except (TypeError, KeyError):
        pass
    reference_kws.setdefault('label', reference_label)
    shroud_kws.setdefault('color', 'k')
    reference_kws.setdefault('color', 'orangered')
    shroud_kws.setdefault('alpha', .02)
    reference_kws.setdefault('linewidth', 1)
    shroud_kws.setdefault('linewidth', .2)
    if common_args == dict():
        common_args = dict(cumulative=True, density=True,
                           histtype='step', bins=100)
    for k, v in common_args.items():
        reference_kws[k] = v
        shroud_kws[k] = v
    for i, simulation in enumerate(simulations):
        if i == len(simulations) - 1:
            shroud_kws.update({'label': shroud_label})
        ax.hist(simulation, **shroud_kws)
    ax.hist(reference, **reference_kws)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax
