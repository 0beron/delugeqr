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
        code, image = qr.deluge_qr("tests/qr.jpg", dbg=False)
        assert code == [0x200aa2ec, 0x200aa3b0, 0x201047a3, 0x2004f30f, 0x9a2b]

    def testqr_rot(self):
        """Test QR ROT"""
        code, image = qr.deluge_qr("tests/qr35.jpg", dbg=False)
        assert code == [0x200aa2ec, 0x200aa3b0, 0x201047a3, 0x2004f30f, 0x9a2b]
        
    def testqr2(self):
        """Test QR2"""
        code, image = qr.deluge_qr("tests/qr2.jpg", dbg=False)
        assert code == [0x200bfadc, 0x200905b0, 0x200dfa70, 0x200bd814, 0x725b]

    def testqr3(self): 
        """Test case QR3"""
        code, image = qr.deluge_qr("tests/qr3.jpg", dbg=False)
        assert code == [0x20070acb, 0x2006a2b9, 0x20080747, 0x20092040, 0xa35a]

    def testqr4(self): 
        """Test case QR4"""
        code, image = qr.deluge_qr("tests/qr4.jpg", dbg=False)
        assert code == [0x20054f25, 0x200931fc, 0x2009dd7c, 0x20047c64, 0xaa55]
        
    def testqr5(self): 
        """Test case QR5"""
        code, image = qr.deluge_qr("tests/qr5.jpg", dbg=False)
        assert code == [0x200e15a4, 0x2009f0c0, 0x200af0b8, 0x20111534, 0xaa55]
        
    def testqr6(self): 
        """Test case QR6"""
        code, image = qr.deluge_qr("tests/qr6.jpg", dbg=False)
        assert code == [0x200931fc, 0x00000000, 0x00000000, 0x00000000, 0xaa55]
        
    def testqr7(self): 
        """Test case QR7"""
        code, image = qr.deluge_qr("tests/qr7.jpg", dbg=False)
        assert code == [0x2008beec, 0x2008be78, 0x201187d0, 0x200df25c, 0x3985]

    def testqr8(self): 
        """Test case QR8"""
        code, image = qr.deluge_qr("tests/qr8.jpg", dbg=False)
        assert code == [0x2008beec, 0x2008be78, 0x201187d0, 0x200df25c, 0x3985]

    def testqr9(self): 
        """Test case QR9"""   
        code, image = qr.deluge_qr("tests/qr9.jpg", dbg=False)
        assert code == [0x2008beec, 0x2008be78, 0x201187d0, 0x200df25c, 0x3985]

    def testqrah(self): 
        """Test case QR3a-h"""
        for ltr in "abcdefgh":
            code, image = qr.deluge_qr(f"tests/qr3{ltr}.jpg", dbg=False)
            assert code == [0x20070acb, 0x2006a2b9, 0x20080747, 0x20092040, 0xa35a]

    def testqr13(self):
        code, image = qr.deluge_qr("tests/qr13.jpg", dbg=False)
        assert code == [0x20059d69, 0x20059d09, 0x20068445, 0x20076bcd, 0xd533]

    def testqr14(self):
        code, image = qr.deluge_qr("tests/qr14.jpg", dbg=False)
        assert code == [0x201152bc, 0x20115358, 0x20110344, 0x20048ecc, 0xd533]

    def testqr16(self):
        code, image = qr.deluge_qr("tests/qr16.jpg", dbg=False)
        assert code == [0x201154e4, 0x20105bd4, 0x20114dcc, 0x2007302f, 0xdd39]

    def testqr17(self):
        code, image = qr.deluge_qr("tests/qr17.jpg", dbg=False)
        assert code == [0x201154e4, 0x20105bd4, 0x20114dcc, 0x2007302f, 0xdd39]

    def testqr18(self):
        code, image = qr.deluge_qr("tests/qr18.jpg", dbg=False)
        assert code == [0x201154e4, 0x20105bd4, 0x2007e599, 0x200ccecc, 0xdd39]

    def testqr19(self):
        code, image = qr.deluge_qr("tests/qr19.jpg", dbg=False)
        assert code == [0x200c1be4, 0x200a86c0, 0x200a55dc, 0x200900e4, 0xfe31]

    def testqr20(self):
        code, image = qr.deluge_qr("tests/qr20.jpg", dbg=False)
        assert code == [0x200dc968, 0x200b8094, 0x2007a56f, 0x2011103d, 0xfe31]

    def testqr21(self):
        code, image = qr.deluge_qr("tests/qr21.jpg", dbg=False)
        assert code == [0x200dc968, 0x200b8094, 0x2007a56f, 0x2011103d, 0x2bce]

        
if __name__ == "__main__":
    unittest.main() # run all tests
