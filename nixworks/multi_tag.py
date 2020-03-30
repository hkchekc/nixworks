import numpy as np
import nixio as nix
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
    start = list()
    end = list()
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


def _sorting(starts, ends):  # li is the start values
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
        if _in_range(st, true_start, true_end):
            true_start = st
        if _in_range(ends[i], true_start, true_end):
            true_end = ends[i]
        # Any one point that is not in range means that there are no intersection
        if not _in_range(st, true_start, true_end) and not \
                _in_range(ends[i], true_start, true_end):
            return None
    if isinstance(true_start, np.ndarray):
        true_slice = tuple([slice(x, y+1) for x, y in zip(true_start, true_end)])
    else:
        true_slice = (slice(true_start, true_end + 1), )
    return nix.data_view.DataView(ref, true_slice)

def _intersect(point, start, end):
    pass


def union(ref, multi_tags):
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
    start_list = []
    end_list = []
    for i, st in enumerate(starts):  # check if any duplicate
        for ti, (tmp_st, tmp_ed) in enumerate(zip(start_list, end_list)):
            if _in_range(st, tmp_st, tmp_ed) or _in_range(ends[i], tmp_st, tmp_ed):
                # The new area is completely covered by old area, ignore!
                if _in_range(st, tmp_st, tmp_ed) and _in_range(ends[i], tmp_st, tmp_ed):
                    continue
                else:
                    # search for intersection
                    # create 1 or 2 new dataview
                    pass
                # if not _in_range(ends[i], tmp_st, tmp_ed):  # ends[i] > tmp_ed
                #     end_list[ti] = ends[i]
                # elif not _in_range(st, tmp_st, tmp_ed):  # st < tmp_st
                #     start_list[ti] = st
            # if not contiguous, then mark as new slices
            else:
                start_list.append(st)
                end_list.append(ends[i])
    view_list = []
    for true_st, true_ed in zip(start_list, end_list):
        true_slice = tuple([slice(x, y+1) for x, y in zip(true_st, true_ed)])
        view_list.append(nix.data_view.DataView(ref, true_slice))
    return view_list
