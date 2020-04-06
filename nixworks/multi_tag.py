import numpy as np
import nixio as nix
import itertools
from nixio import exceptions, data_view


def _check_valid(multi_tags, ref):
    for mt in multi_tags:
        if not (isinstance(mt, nix.MultiTag) or isinstance(mt, nix.Tag)):
            raise TypeError("Input must be either MultiTags or Tags.")
    for mt in multi_tags:
        if ref not in mt.references and not isinstance(ref, int):
            raise ValueError("This DataArray is not referenced.")
        if isinstance(ref, int) and ref > len(mt.references):
            raise nix.exceptions.OutOfBounds("The index given is out of bound.")


def _populate_start_end(multi_tags):
    start = []
    end = []
    for mt in multi_tags:
        if isinstance(mt, nix.MultiTag):
            start.extend(list(mt.positions))
            if mt.extents is not None:
                end.extend([e+p for e, p in zip(mt.extents[:], mt.positions[:])])
            else:
                end.extend(mt.positions)
        else:
            start.append(mt.position)
            if mt.extent is not None:
                e = [e+p for e, p in zip(mt.extent, mt.position)]
                end.append(e)
    start = np.array(start, dtype=int)
    end = np.array(end, dtype=int)
    if start.shape != end.shape:
        raise nix.exceptions.IncompatibleDimensions("Start and End position "
                                                    "shapes do not match.", "")
    return start, end


def _in_range(point, start, end):
    # loop for each dimension
    if isinstance(start, np.ndarray):
        for i, st in enumerate(start):
            if point[i] < st or point[i] > end[i]:
                    return False
    else:
        if point < start or point > end:
            return False
    return True


def _sorting(starts, ends):
    starts = starts.tolist()
    ends = ends.tolist()
    sort = [i for i in sorted(enumerate(starts), key=lambda s: s[1])]
    sorted_starts = np.array([s[1] for s in sort])
    sorted_ends = np.array([ends[s[0]] for s in sort])
    return sorted_starts, sorted_ends


def intersection(ref, multi_tags):
    """
    Function to return the overlapping area in a specified DataArray tagged by multiple Tags/MultiTags.
    :param ref: the referenced array
    :param multi_tags: Tags or MultiTags that point to the tagged data
    :return: a DataView
    """
    _check_valid(multi_tags, ref)
    if not isinstance(ref, nix.DataArray):
        ref = multi_tags[0].references[ref]
    starts, ends = _populate_start_end(multi_tags)
    true_start = starts[0]
    true_end = ends[0]
    for j, st in enumerate(starts[1:]):
        i = j + 1
        true_start, true_end = _intersect(st, ends[i], true_start, true_end)
        if true_start is None:  # true_end must also be None
            return None
    if isinstance(true_start, np.ndarray):
        true_slice = tuple([slice(x, y+1) for x, y in zip(true_start, true_end)])
    else:
        true_slice = (slice(true_start, true_end + 1), )
    return nix.data_view.DataView(ref, true_slice)


def _intersect(st, ed, true_start, true_end):
    # Internal function to check for intersection and return its start and end
    if _in_range(st, true_start, true_end):
        true_start = st
    if _in_range(ed, true_start, true_end):
        true_end = ed
    #  check if other corners maybe in range
    if not _in_range(st, true_start, true_end) and not \
            _in_range(ed, true_start, true_end):
        if isinstance(st, np.ndarray):
            tmp_starts = [(x, y) for x, y in zip(st, true_start)]
            tmp_ends = [(x, y) for x, y in zip(ed, true_end)]
        else:
            tmp_starts = [(st, true_start)]
            tmp_ends = [(ed, true_end)]
        new_starts = itertools.product(*tmp_starts)
        new_ends = itertools.product(*tmp_ends)
        # based on the (probably true) assumption that only one such point in range
        start_update = False
        end_update = False
        for news in new_starts:
            if _in_range(news, true_start, true_end) and _in_range(news, st, ed):
                start_update = True
                tmp_start = news
                break
        for newe in new_ends:
            if _in_range(newe, true_start, true_end) and _in_range(newe, st, ed):
                end_update = True
                true_end = newe
                break
        if not start_update and not end_update:
            return None, None
        elif start_update + end_update == 1:
            raise ValueError("Only start or end point of intersection exists.")
        elif start_update and end_update:
            true_start = tmp_start
    return true_start, true_end


def group_slices(st, ed, start_list, end_list):
    inter_idx = None
    for grp_i, (grp_st, grp_ed) in enumerate(zip(start_list, end_list)):
        for single_st, single_ed in zip(grp_st, grp_ed):
            # The new area is completely covered by old area, ignore
            if _in_range(st, single_st, single_ed) and _in_range(ed, single_st, single_ed):
                return start_list, end_list
            check_intersect = _intersect(st, ed, single_st, single_ed)
            if check_intersect[0] is None:
                # create new group
                start_list.append([st])
                start_list.append([ed])
            else:
                if inter_idx is None:  # there is an intersection
                    inter_idx = grp_i
                    start_list[grp_i].append(st)
                    end_list[grp_i].append(ed)
                    break # break one loop, start from next group
                else:  # there are previous intersection with this st, ed pair
                    # merge the previous intersected group and current group
                    # note that the current (st, ed) is already in the previous grp
                    start_list[inter_idx].extend(grp_st)
                    end_list[inter_idx].extend(grp_ed)
                    # pop the current group
                    del start_list[grp_i]
                    del end_list[grp_i]
    return start_list, end_list



def union(ref, multi_tags, dimension_prior=0):
    """
    Function to return the (non-overlapping) union of area tagged by multiple Tags
    or MultiTags of a specified DataArray.
    :param ref: the referenced array
    :param multi_tags: Tags or MultiTags that point to the tagged data
    :return: a list of DataViews
    """
    _check_valid(multi_tags, ref)
    if not isinstance(ref, nix.DataArray):
        ref = multi_tags[0].references[ref]
    starts, ends = _populate_start_end(multi_tags)
    starts, ends = _sorting(starts, ends)
    start_list = [[starts[0]]]
    end_list = [[ends[0]]]
    for j, st in enumerate(starts[1:]):
        i = j+1
        start_list, end_list = group_slices(st, ends[i], start_list, end_list)
    

    view_list = []
    for true_st, true_ed in zip(start_list, end_list):
        true_slice = tuple([slice(x, y+1) for x, y in zip(true_st, true_ed)])
        view_list.append(nix.data_view.DataView(ref, true_slice))
    return view_list
