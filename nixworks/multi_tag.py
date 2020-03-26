import numpy as np
import nixio as nix


def _check_valid(multi_tags, ref):
    for mt in multi_tags:
        if not (isinstance(mt, nix.MultiTag) or isinstance(mt, nix.Tag)):
            raise TypeError("Input must be either MultiTags or Tags.")
    for mt in multi_tags:
        if ref not in mt.references:
            raise ValueError("This DataArray is not referenced.")


def _populate_start_end(multi_tags):
    start = list()
    end = list()
    for mt in multi_tags:
        if isinstance(mt, nix.MultiTag):
            start.extend(list(mt.positions))
            if mt.extents is not None:
                ext = mt.extents[:]
                end.extend([e+p for e,p in zip(ext, mt.positions[:])])
            else:
                end.extend(list(mt.positions))
        else:
            start.append(mt.position)
            if mt.extent is not None:
                e = [e+p for e,p in zip(mt.extent, mt.position)]
                end.append(e)
    start = np.array(start, dtype=int)
    end = np.array(end, dtype=int)
    return start, end


def _in_range(point, start, end):
    for i, st in enumerate(start):
        if point[i] < st or point[i] > end[i]:
                return False
    return True


def _sorting(li):  # li is the start values
    if len(np.array(li).shape) == 1:
        sort_idx = np.argsort(li)
        sorted_starts = sorted(li)
    else: # sort only by first dimension
        sort_idx = np.argsort(np.transpose(li)[0])
        sorted_starts = sorted(li, key=lambda l: li[0])
    return sort_idx, sorted_starts


def union(multi_tags, ref):
    # now the simple case of 2 tags
    _check_valid(multi_tags, ref)
    if not isinstance(ref, nix.DataArray):
        ref = multi_tags[0].references[ref]
    starts, ends = _populate_start_end(multi_tags)
    for i, st in enumerate(starts):


    return nix.data_view.DataView(ref, None)

def intersection(ref, multi_tags):
    _check_valid(multi_tags, ref)
    if not isinstance(ref, nix.DataArray):
        ref = multi_tags[0].references[ref]
    starts, ends = _populate_start_end(multi_tags)
    true_start = starts[0]
    true_end = ends[0]
    for j, st in enumerate(starts[1:]):
        i = j + 1
        if _in_range(st, true_start, true_end):
            true_start = st
        if _in_range(ends[i], true_start, true_end):
            true_end = ends[i]
        # Any one point that is not in range means that there are no intersection
        if not _in_range(st, true_start, true_end) and not \
                _in_range(ends[i], true_start, true_end):
            return None
    true_slice = tuple([slice(x, y+1) for x, y in zip(true_start, true_end)])
    return nix.data_view.DataView(ref, true_slice)

