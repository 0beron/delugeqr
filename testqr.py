import unittest
import qr


class SimpleTestCase(unittest.TestCase):

    def setUp(self):
        """Call before every test case."""
        pass

    def tearDown(self):
        """Call after every test case."""
        pass

    def testqr(self):
        """Test QR"""
        assert qr.deluge_qr("qr.jpg", dbg=0) == [0x200aa2ec, 0x200aa3b0, 0x201047a3, 0x2004f30f, 0x9a2b]

    def testqr_rot(self):
        """Test QR ROT"""
        assert qr.deluge_qr("qr35.jpg", dbg=0) == [0x200aa2ec, 0x200aa3b0, 0x201047a3, 0x2004f30f, 0x9a2b]
        
    def testqr2(self):
        """Test QR2"""
        assert qr.deluge_qr("qr2.jpg", dbg=0) == [0x200bfadc, 0x200905b0, 0x200dfa70, 0x200bd814, 0x725b]

    def testqr3(self): 
        """Test case QR3"""
        assert qr.deluge_qr("qr3.jpg", dbg=0) == [0x20070acb, 0x2006a2b9, 0x20080747, 0x20092040, 0xa35a]

    def testqr4(self): 
        """Test case QR4"""
        assert qr.deluge_qr("qr4.jpg", dbg=0) == [0x20054f25, 0x200931fc, 0x2009dd7c, 0x20047c64, 0xaa55]
        
    def testqr5(self): 
        """Test case QR5"""
        assert qr.deluge_qr("qr5.jpg", dbg=0) == [0x200e15a4, 0x2009f0c0, 0x200af0b8, 0x20111534, 0xaa55]
        
    def testqr6(self): 
        """Test case QR6"""
        assert qr.deluge_qr("qr6.jpg", dbg=0) == [0x200931fc, 0x00000000, 0x00000000, 0x00000000, 0xaa55]
        
    def testqr7(self): 
        """Test case QR7"""
        assert qr.deluge_qr("qr7.jpg", dbg=0) == [0x2008beec, 0x2008be78, 0x201187d0, 0x200df25c, 0x3985]
        assert qr.deluge_qr("qr8.jpg", dbg=0) == [0x2008beec, 0x2008be78, 0x201187d0, 0x200df25c, 0x3985]
        assert qr.deluge_qr("qr9.jpg", dbg=0) == [0x2008beec, 0x2008be78, 0x201187d0, 0x200df25c, 0x3985]

    def testqrah(self): 
        """Test case QR3a-h"""
        for ltr in "abcdefgh":
            assert qr.deluge_qr(f"qr3{ltr}.jpg", dbg=0) == [0x20070acb, 0x2006a2b9, 0x20080747, 0x20092040, 0xa35a]
        
if __name__ == "__main__":
    unittest.main() # run all tests
