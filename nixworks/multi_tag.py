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
                end.extend([e+p for e,p in zip(mt.extents[:], mt.positions[:])])
            else:
                end.extend(mt.positions)
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


def _sorting(starts, ends):  # li is the start values
    sort = [i for i in sorted(enumerate(starts), key=lambda s: s[1])]
    sorted_starts = np.array([s[1] for s in sort])
    sorted_ends = np.array([ends[s[0]] for s in sort])
    return sorted_starts, sorted_ends


def union(ref, multi_tags):
    # now the simple case of 2 tags
    _check_valid(multi_tags, ref)
    if not isinstance(ref, nix.DataArray):
        ref = multi_tags[0].references[ref]
    starts, ends = _populate_start_end(multi_tags)
    starts, ends = _sorting(starts, ends)
    start_list = []
    end_list = []
    for i, st in enumerate(starts):  # check if any duplicate
        covered = False
        for ti, tmp_st, tmp_ed in enumerate(zip(start_list, end_list)):
            if _in_range(st, tmp_st, tmp_ed) or _in_range(ends[i], tmp_st, tmp_ed):
                covered = True
                if not _in_range(ends[i], tmp_st, tmp_ed):  # ends[i] > tmp_ed
                    end_list[ti] = ends[i]
                elif not _in_range(st, tmp_st, tmp_ed):  # st < tmp_st
                    start_list[ti] = st
        # if not contiguous, then mark as new slices
        if not covered:
            start_list.append(st)
            end_list.append(ends[i])
    view_list = []
    for true_st, true_ed in zip(start_list, end_list):
        true_slice = tuple([slice(x, y+1) for x, y in zip(true_st, true_ed)])
        view_list.append(nix.data_view.DataView(ref, true_slice))
    return view_list


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

