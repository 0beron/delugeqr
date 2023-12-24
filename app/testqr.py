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
        code, image = qr.deluge_qr("../tests/qr.jpg", dbg=False)
        assert code == [0x200AA2EC, 0x200AA3B0, 0x201047A3, 0x2004F30F, 0x9A2B]

    def testqr_rot(self):
        """Test QR ROT"""
        code, image = qr.deluge_qr("../tests/qr35.jpg", dbg=False)
        assert code == [0x200AA2EC, 0x200AA3B0, 0x201047A3, 0x2004F30F, 0x9A2B]

    def testqr2(self):
        """Test QR2"""
        code, image = qr.deluge_qr("../tests/qr2.jpg", dbg=False)
        assert code == [0x200BFADC, 0x200905B0, 0x200DFA70, 0x200BD814, 0x725B]

    def testqr3(self):
        """Test case QR3"""
        code, image = qr.deluge_qr("../tests/qr3.jpg", dbg=False)
        assert code == [0x20070ACB, 0x2006A2B9, 0x20080747, 0x20092040, 0xA35A]

    def testqr4(self):
        """Test case QR4"""
        code, image = qr.deluge_qr("../tests/qr4.jpg", dbg=False)
        assert code == [0x20054F25, 0x200931FC, 0x2009DD7C, 0x20047C64, 0xAA55]

    def testqr5(self):
        """Test case QR5"""
        code, image = qr.deluge_qr("../tests/qr5.jpg", dbg=False)
        assert code == [0x200E15A4, 0x2009F0C0, 0x200AF0B8, 0x20111534, 0xAA55]

    def testqr6(self):
        """Test case QR6"""
        code, image = qr.deluge_qr("../tests/qr6.jpg", dbg=False)
        assert code == [0x200931FC, 0x00000000, 0x00000000, 0x00000000, 0xAA55]

    def testqr7(self):
        """Test case QR7"""
        code, image = qr.deluge_qr("../tests/qr7.jpg", dbg=False)
        assert code == [0x2008BEEC, 0x2008BE78, 0x201187D0, 0x200DF25C, 0x3985]

    def testqr8(self):
        """Test case QR8"""
        code, image = qr.deluge_qr("../tests/qr8.jpg", dbg=False)
        assert code == [0x2008BEEC, 0x2008BE78, 0x201187D0, 0x200DF25C, 0x3985]

    def testqr9(self):
        """Test case QR9"""
        code, image = qr.deluge_qr("../tests/qr9.jpg", dbg=False)
        assert code == [0x2008BEEC, 0x2008BE78, 0x201187D0, 0x200DF25C, 0x3985]

    def testqrah(self):
        """Test case QR3a-h"""
        for ltr in "abcdefgh":
            code, image = qr.deluge_qr(f"../tests/qr3{ltr}.jpg", dbg=False)
            assert code == [0x20070ACB, 0x2006A2B9, 0x20080747, 0x20092040, 0xA35A]

    def testqr13(self):
        code, image = qr.deluge_qr("../tests/qr13.jpg", dbg=False)
        assert code == [0x20059D69, 0x20059D09, 0x20068445, 0x20076BCD, 0xD533]

    def testqr14(self):
        code, image = qr.deluge_qr("../tests/qr14.jpg", dbg=False)
        assert code == [0x201152BC, 0x20115358, 0x20110344, 0x20048ECC, 0xD533]

    def testqr16(self):
        code, image = qr.deluge_qr("../tests/qr16.jpg", dbg=False)
        assert code == [0x201154E4, 0x20105BD4, 0x20114DCC, 0x2007302F, 0xDD39]

    def testqr17(self):
        code, image = qr.deluge_qr("../tests/qr17.jpg", dbg=False)
        assert code == [0x201154E4, 0x20105BD4, 0x20114DCC, 0x2007302F, 0xDD39]

    def testqr18(self):
        code, image = qr.deluge_qr("../tests/qr18.jpg", dbg=False)
        assert code == [0x201154E4, 0x20105BD4, 0x2007E599, 0x200CCECC, 0xDD39]

    def testqr19(self):
        code, image = qr.deluge_qr("../tests/qr19.jpg", dbg=False)
        assert code == [0x200C1BE4, 0x200A86C0, 0x200A55DC, 0x200900E4, 0xFE31]

    def testqr20(self):
        code, image = qr.deluge_qr("../tests/qr20.jpg", dbg=False)
        assert code == [0x200DC968, 0x200B8094, 0x2007A56F, 0x2011103D, 0xFE31]

    def testqr21(self):
        code, image = qr.deluge_qr("../tests/qr21.jpg", dbg=False)
        assert code == [0x200DC968, 0x200B8094, 0x2007A56F, 0x2011103D, 0x2BCE]

    def testqr22(self):
        code, image = qr.deluge_qr("../tests/qr22.jpg", dbg=False)
        assert code == [0X2013B0B7, 0X2004925C, 0X2004710C, 0X00000000, 0XDE88]

    def testqr23(self):
        code, image = qr.deluge_qr("../tests/qr23.jpg", dbg=False)
        assert code == [0X200AA500, 0X200AA668, 0X20104BDB, 0X2004F37F, 0XFE31]

    def testqr24(self):
        code, image = qr.deluge_qr("../tests/qr24.jpg", dbg=False)
        assert code == [0X20115848, 0X200DAAD4, 0X20105784, 0X2011589C, 0XFE31]

    def testqr25(self):
        code, image = qr.deluge_qr("../tests/qr25.jpg", dbg=False)
        assert code == [0X20115848, 0X200DAAD4, 0X20105784, 0X2011589C, 0XFE31]

    def testqr26(self):
        code, image = qr.deluge_qr("../tests/qr26.jpg", dbg=False)
        assert code == [0X200DC968, 0X200B8094, 0X2007A56F, 0X2011103D, 0XFE31]

    def testqr27(self):
        code, image = qr.deluge_qr("../tests/qr27.png", dbg=False)
        assert code == [0X20096A04, 0X20082F00, 0X2004CCDC, 0X20044F88, 0XFE31]

    def testqr28(self):
        code, image = qr.deluge_qr("../tests/qr28.jpg", dbg=False)
        assert code == [0x20051543, 0x201217FA, 0x2011AD34, 0x2004FB47, 0xFA10]

    def testqr29(self):
        code, image = qr.deluge_qr("../tests/qr29.jpg", dbg=False)
        assert code == [0x20119B05, 0x20119C27, 0x200FF423, 0x20098C8C, 0x7145]

if __name__ == "__main__":
    unittest.main()  # run all tests
