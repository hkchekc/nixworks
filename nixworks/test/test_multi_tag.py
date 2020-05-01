
import unittest
from ..multi_tag import intersection, union
from tempfile import mkdtemp
import os
import shutil
import numpy as np
import nixio as nix


class TestMultiTag(unittest.TestCase):

    def setUp(self):
        self.p = mkdtemp()
        self.testfilename = os.path.join(self.p, "test.nix")
        self.file = nix.File.open(self.testfilename, nix.FileMode.Overwrite)
        self.block = self.file.create_block("test_block", "abc")
        self.arr1d = np.arange(1000)
        self.ref1d = self.block.create_data_array("test1d", "test", data=self.arr1d)
        self.ref1d.append_set_dimension()
        self.arr3d = np.arange(1000).reshape((10, 10, 10))
        self.ref3d = self.block.create_data_array("test3d", "test", data=self.arr3d)
        self.ref3d.append_set_dimension()
        self.ref3d.append_set_dimension()

    def tearDown(self):
        self.file.close()
        shutil.rmtree(self.p)

    def test_two_1d(self):
        t1 = self.block.create_tag("t1", "test", position=(0,))
        t2 = self.block.create_tag("t2", "test", position=(10,))
        t1.references.append(self.ref1d)
        t2.references.append(self.ref1d)

        # detached - no intersection
        t1.extent = [5]
        t2.extent = [5]
        i = intersection(self.ref1d, [t1, t2])
        assert i is None

        # intersected interval
        t1.extent = [12]
        i = intersection(self.ref1d, [t1, t2])
        np.testing.assert_array_almost_equal(np.array(i), self.arr1d[10:13])

        # intersected point
        t1.extent = [10]
        i = intersection(self.ref1d, [t1, t2])
        np.testing.assert_array_almost_equal(np.array(i), self.arr1d[10:13])
        # covered
        t1.extent = [30]
        i = intersection(self.ref1d, [t1, t2])
        np.testing.assert_array_almost_equal(np.array(i), t2.tagged_data(0)[:])

    def test_2d(self):
        # create a 2d reference data
        ref_data = np.arange(100).reshape((10, 10))
        ref2d = self.block.create_data_array('ref2d', 'ref', data=ref_data)
        # There should be six different styles of 2d tags intersecting
        # 1. tags that intersects at the corner
        t1 = self.block.create_tag("corner1", "test", position=(0, 0))
        t2 = self.block.create_tag("corner2", "test", position=(5, 5))
        t1.references.append(ref2d)
        t2.references.append(ref2d)
        t1.extent = [7, 7]
        t2.extent = [4, 4]
        i = intersection(ref2d, [t1, t2])
        np.testing.assert_array_almost_equal(np.array(i), ref2d[5:8, 5:8])
        # 2. tags that intersects as a cross form
        t1 = self.block.create_tag("cross1", "test", position=(5, 0))
        t2 = self.block.create_tag("cross2", "test", position=(2, 2))
        t1.references.append(ref2d)
        t2.references.append(ref2d)
        t1.extent = [2, 8]  # end point is (7, 8)
        t2.extent = [6, 2]  # end point is (8, 4)
        i = intersection(ref2d, [t1, t2])
        np.testing.assert_array_almost_equal(np.array(i), ref2d[5:8, 2:5])
        # 3. one tag is completely covered by a larger tag
        t1 = self.block.create_tag("inclusive1", "test", position=(0, 0))
        t2 = self.block.create_tag("inclusive2", "test", position=(2, 2))
        t1.references.append(ref2d)
        t2.references.append(ref2d)
        t1.extent = [7, 7]
        t2.extent = [3, 3]
        i = intersection(ref2d, [t1, t2])
        np.testing.assert_array_almost_equal(np.array(i), ref2d[2:6, 2:6])
        # 4. tags that intersect as a rectangle
        t1 = self.block.create_tag("rect1", "test", position=(0, 0))
        t2 = self.block.create_tag("rect2", "test", position=(0, 2))
        t1.references.append(ref2d)
        t2.references.append(ref2d)
        t1.extent = [4, 4]
        t2.extent = [4, 4]
        i = intersection(ref2d, [t1, t2])
        np.testing.assert_array_almost_equal(np.array(i), ref2d[0:5, 2:5])
        # 5. T-shape: like a cross, but only stick out on one side
        t1 = self.block.create_tag("t-shape1", "test", position=(0, 0))
        t2 = self.block.create_tag("t-shape2", "test", position=(2, 2))
        t1.references.append(ref2d)
        t2.references.append(ref2d)
        t1.extent = [6, 6]
        t2.extent = [5, 3]  # end point (7, 5)  #TODO: BUG
        i = intersection(ref2d, [t1, t2])
        np.testing.assert_array_almost_equal(np.array(i), ref2d[2:7, 2:6])
        # 6. no intersection
        t1 = self.block.create_tag("notinter1", "test", position=(0, 0))
        t2 = self.block.create_tag("notinter2", "test", position=(5, 5))
        t1.references.append(ref2d)
        t2.references.append(ref2d)
        t1.extent = [3, 3]
        t2.extent = [3, 3]
        i = intersection(ref2d, [t1, t2])
        assert i is None

    def test_multi_nd(self):
        d = np.zeros((2, 3))
        d[1] += 1
        p1 = self.block.create_data_array("pos1", "pos", data=d)
        t1 = self.block.create_multi_tag("t1", "test", positions=p1)
        e1 = self.block.create_data_array("ext1", "ext", data=np.ones((2, 3))*3)
        t1.extents = e1
        ###################################################################
        a = np.array([[2, 2, 2], [3, 3, 3]])
        p2 = self.block.create_data_array("pos2", "pos", data=a)
        t2 = self.block.create_multi_tag("t2", "test", positions=p2)
        e2 = self.block.create_data_array("ext2", "ext", data=np.ones((2, 3))*2)
        t2.extents = e2
        t1.references.append(self.ref3d)
        t2.references.append(self.ref3d)
        # intersections
        i = intersection(self.ref3d, [t1])
        np.testing.assert_array_almost_equal(i[:], self.arr3d[1:4, 1:4, 1:4])
        i = intersection(self.ref3d, [t1, t2])
        np.testing.assert_array_almost_equal(i[:], self.arr3d[3, 3, 3])
        # union
        u = union(self.ref3d, [t1])
        np.testing.assert_array_almost_equal(u[0][:], self.arr3d[0:5, 0:5, 0:5])
        u = union(self.ref3d, [t1, t2])
        np.testing.assert_array_almost_equal(u[0][:], self.arr3d[0:6, 0:6, 0:6])
        p3 = self.block.create_data_array("pos3", "pos", data=np.array([[5, 5, 5], [6, 6, 6]]))
        t3 = self.block.create_multi_tag("t3", "test", positions=p3)
        e3 = self.block.create_data_array("ext3", "ext", data=np.ones((2, 3)) * 3)
        t3.extents = e3
        t3.references.append(self.ref3d)
        u = union(self.ref3d, [t1, t3])
        np.testing.assert_array_almost_equal(u[0][:], self.arr3d[:, :, :])
        np.testing.assert_array_almost_equal(u[0][:], self.arr3d[:, :, :])

    def test_flat_positions(self):
        # 1d position and extent
        pos3 = self.block.create_data_array("pos3", "pos", data=np.array([1, 0, 0]))
        pos4 = self.block.create_data_array("pos4", "pos", data=np.array([0, 1, 1]))
        ext3 = self.block.create_data_array("ext3", "ext", data=np.array([2, 2, 2]))
        ext4 = self.block.create_data_array("ext4", "ext", data=np.array([2, 2, 2]))
        t3 = self.block.create_multi_tag("t3", "test", positions=pos3)
        t3.extents = ext3
        t4 = self.block.create_multi_tag("t4", "test", positions=pos4)
        t4.extents = ext4
        t3.references.append(self.ref1d)
        t4.references.append(self.ref1d)
        intersection(0, [t3, t4])

    def test_union(self):
        pass
        # u = union(self.ref1d, [t1, t2])
        # np.testing.assert_array_equal(u[0], self.arr1d[0:6])
        # np.testing.assert_array_equal(u[1], self.arr1d[10:16])
        # u = union(self.ref1d, [t1, t2])
        # np.testing.assert_array_equal(u[0], self.arr1d[0:16])
        # u = union(self.ref1d, [t1, t2])
        # np.testing.assert_array_equal(u[0], t1.tagged_data(0)[:])

