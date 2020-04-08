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
        new_starts = itertools.product(*tmp_starts)  # all corners of the range
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


def _group_slices(st, ed, start_list, end_list, intersect_list):
    inter_idx = None
    for grp_i, (grp_st, grp_ed) in enumerate(zip(start_list, end_list)):
        for single_st, single_ed in zip(grp_st, grp_ed):
            # The new area is completely covered by old area, ignore
            if _in_range(st, single_st, single_ed) and _in_range(ed, single_st, single_ed):
                intersect_list.append((None, None))
                return start_list, end_list, intersect_list
            check_intersect = _intersect(st, ed, single_st, single_ed)
            if check_intersect[0] is None:
                # create new group
                start_list.append([st])
                start_list.append([ed])
                intersect_list.append([None])
            else:
                if inter_idx is None:  # there is an intersection but no previous intersections
                    inter_idx = grp_i
                    start_list[grp_i].append(st)
                    end_list[grp_i].append(ed)
                    intersect_list[grp_i] = [check_intersect]
                    break  # break one loop, start from next group
                else:  # there are previous intersection with this st, ed pair
                    # merge the previous intersected group and current group
                    # note that the current (st, ed) is already in the previous grp
                    start_list[inter_idx].extend(grp_st)
                    end_list[inter_idx].extend(grp_ed)
                    intersect_list[inter_idx].append([check_intersect])  # TODO: need test if all case is covered
                    # pop the current group
                    del start_list[grp_i]
                    del end_list[grp_i]
                    del intersect_list[grp_i]
    return start_list, end_list, intersect_list


def union(ref, multi_tags, dimension_prior=0):
    """
    Function to return the (non-overlapping) union of area tagged by multiple Tags
    or MultiTags of a specified DataArray.
    :param ref: the referenced array
    :param multi_tags: Tags or MultiTags that point to the tagged data
    :param dimension_prior: pass
    :return: a list of DataViews
    """
    _check_valid(multi_tags, ref)
    if not isinstance(ref, nix.DataArray):
        ref = multi_tags[0].references[ref]
    starts, ends = _populate_start_end(multi_tags)
    starts, ends = _sorting(starts, ends)
    start_list = [[starts[0]]]
    end_list = [[ends[0]]]
    intersect_list = []
    for j, st in enumerate(starts[1:]):
        i = j+1
        start_list, end_list, intersect_list = _group_slices(st, ends[i], start_list, end_list, intersect_list)
    # start_list, end_list should have same shape
    # create the DataViews
    view_list = []
    for gi, (st_grp, ed_grp) in enumerate(zip(start_list, end_list)):
        if len(st_grp) == 1:
            true_slice = tuple([slice(x, y + 1) for x, y in zip(st_grp[0], ed_grp[0])])
            view_list.append(nix.data_view.DataView(ref, true_slice))
        elif len(st_grp) == 2:
            if _is_rectangle(st_grp[0], ed_grp[0], st_grp[1], ed_grp[1]):
                true_slice = tuple([slice(x, y + 1) for x, y in zip(st_grp[0], ed_grp[1])])
                view_list.append(nix.data_view.DataView(ref, true_slice))
            else:
                inter = intersect_list[gi][0]  # just one intersection as there are only 2

        else:
            combinations = itertools.combinations(range(len(st_grp)), 2)
            for combi in combinations:
                if _is_rectangle(st_grp[combi[0]], ed_grp[combi[0]], st_grp[combi[1]], ed_grp[combi[1]]):
                    end_list[gi][combi[0]] = ed_grp[combi[1]]
                    del start_list[gi][combi[1]]
                    del end_list[gi][combi[1]]
                    del intersect_list[gi][combi[1]-1]  # combi[1] will never be 0 as it is sorted
            if len(start_list[gi]) == 1:  # everything in group combined to one rectangle
                true_slice = tuple([slice(x, y + 1) for x, y in zip(st_grp[0], ed_grp[0])])
                view_list.append(nix.data_view.DataView(ref, true_slice))
            else:
                pass
    return view_list


def _is_rectangle(st1, ed1, st2, ed2):
    st_mask = [i == j for i, j in zip(st1, st2)]
    if sum(st_mask) >= len(st1) - 1:  # only one False
        ed_mask = [i == j for i, j in zip(ed1, ed2)]
        if st_mask == ed_mask:
            return True
    return False
