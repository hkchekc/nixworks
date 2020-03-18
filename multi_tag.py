import numpy as np
import nixio as nix


def _check_valid(multi_tags):
    for mt in multi_tags:
        if not (isinstance(mt, nix.MultiTag) or isinstance(mt, nix.Tag)):
            raise TypeError("Input must be either MultiTags or Tags.")


def _populate_start_end(multi_tags):
    start = list()
    end = list()
    for mt in multi_tags:
        if isinstance(mt, nix.MultiTag):
            start.extend(list(mt.positions))
            if mt.extents is not None:
                ext = list(mt.extents[:])
                end.extend([e+p for e,p in zip(ext, mt.positions)])
            else:
                end.extend(list(mt.positions))
        else:
            start.append(mt.position)
            if mt.extent is not None:
                e = mt.position + mt.extent
                end.append(e)

    return start, end


def union(multi_tags):
    # now the simple case of 2 tags
    _check_valid(multi_tags)
    starts, ends = _populate_start_end(multi_tags)
    start_sort_idx = np.argsort(starts)
    sorted_start = sorted(starts)



def intersection(multi_tags):
    _check_valid(multi_tags)
    start, end = _populate_start_end(multi_tags)


