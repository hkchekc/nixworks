import numpy as np
import nixio as nix
import unittest
from ..multi_tag import *
from tempfile import mkdtemp
import os
import shutil


class TestMultiTag(unittest.TestCase):

    def setUp(self):
        self.p = mkdtemp()
        self.testfilename = os.path.join(self.p, "test.nix")
        self.file = nix.File.open(self.testfilename, nix.FileMode.Overwrite)

        self.block = self.file.create_block("test_block", "abc")
        self.arr1d = np.arange(1000)
        self.ref1d = self.block.create_data_array("test1d", "test", data=self.arr1d)
        self.ref1d.append_set_dimension()
        self.arr3d = np.arange(1000).reshape((10,10,10))
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
        i = intersection(self.ref1d, [t1,t2])
        assert i is None
        # intersected
        t1.extent = [12]
        i = intersection(self.ref1d, [t1,t2])
        np.testing.assert_array_almost_equal(np.array(i), self.arr1d[10:13])
        # covered
        t1.extent = [30]
        i = intersection(self.ref1d, [t1,t2])
        np.testing.assert_array_almost_equal(np.array(i), t2.tagged_data(0)[:])

    def test_multi_nd(self):
        d = np.zeros((2, 3))
        d[1] += 1
        p1 = self.block.create_data_array("pos1", "pos", data=d)
        a = np.array([[2, 2, 2], [3, 3, 3]])
        p2 = self.block.create_data_array("pos2", "pos", data=a)
        t1 = self.block.create_multi_tag("t1", "test", positions=p1)
        t2 = self.block.create_multi_tag("t2", "test", positions=p2)
        e1 = self.block.create_data_array("ext1", "ext", data=np.ones((2, 3))*3)
        e2 = self.block.create_data_array("ext2", "ext", data=np.ones((2, 3))*2)
        t1.extents = e1
        t2.extents = e2
        t1.references.append(self.ref3d)
        t2.references.append(self.ref3d)
        # intersections
        i = intersection(self.ref3d, [t1])
        np.testing.assert_array_almost_equal(i[:], self.arr3d[1:4, 1:4, 1:4])
        i = intersection(self.ref3d, [t1,t2])
        np.testing.assert_array_almost_equal(i[:], self.arr3d[3,3,3])
        # union

