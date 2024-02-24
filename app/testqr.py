import unittest
import argparse
import os
import qr

methods = ['BOTH', 'GRID', 'HOUGH']

class ParametrizedTestCase(unittest.TestCase):
    """ TestCase classes that want to be parametrized should
        inherit from this class.
    """
    def __init__(self, methodName='runTest', **kwargs):
        super(ParametrizedTestCase, self).__init__(methodName)
        self.__dict__.update(kwargs)
        #self.method = method
        #self.filename = filename
        #self.code = code
        #self.rebaseline = rebaseline

    @staticmethod
    def parametrize(testcase_klass, **kwargs):
        """ Create a suite containing all tests taken from the given
            subclass, passing them the parameter 'param'.
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_klass)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_klass(name, **kwargs))
        return suite


def format_code(code):
    if code is None:
        return "None"
    cx = " ".join([f"{x:08x}" for x in code[:4]])
    cy = f"{code[4]:04x}"
    return f"{cx} {cy}"


class QRTestCase(ParametrizedTestCase):
    def setUp(self):
        """Call before every test case."""
        print(f"{self.filename, methods[self.method]}")

    def tearDown(self):
        if self.rebaseline:
            statuses = {ord('p'):"PASS", ord('f'):"FAIL", ord('x'):"XFAIL", ord('m'):"MATCH"}
            if self.keystroke in statuses:
                status = statuses[self.keystroke]
            else:
                status = "UNKNOWN"
            self.outputfp.write(f"{self.filename} {methods[self.method]} {status} {format_code(self.found_code)}\n")

    def testqr(self):
        """Test QR"""
        if self.rebaseline:
            dbg = qr.REBASELINE
        else:
            dbg = qr.NONE
        self.found_code, self.image, self.keystroke = \
            qr.deluge_qr(f"../tests/{self.filename}", method=self.method, dbg=dbg, expected=self.code)
        self.assertEqual(self.found_code, self.code, self.failMessage())

    def failMessage(self):
        return f"\n{self.filename, methods[self.method]}\n" + \
            f"Found:    {format_code(self.found_code)}\n" + \
            f"Expected: {format_code(self.code)}"

    
def read_codes():
    allcodes = {}
    with open("codes.txt", "r") as fp:
        for line in fp:
            bits = line.split()
            status = bits[2]
            allcodes[(bits[0], bits[1])] = [int(hx,16) for hx in bits[3:]]

    return allcodes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DelugeQR Test Program")
    
    parser.add_argument('-r', '--rebaseline', action='store_true', help="Rebaseline")
    parser.add_argument('-n', '--number', default="", type=str, help="Single test to run, eg, 7 or, 3f")
    parser.add_argument('-m', '--method', choices=['grid', 'hough', 'both'],
                        default='both', help='Choose grid detection method')

    args = parser.parse_args()
    method_map = {'both':[qr.GRID, qr.HOUGH], 'grid':[qr.GRID], 'hough':[qr.HOUGH]}

    methodlist = method_map[args.method]

    names = os.listdir("../tests")
    
    names.sort(key=lambda f:int(''.join(filter(lambda c:c.isdigit(), f))))

    expected_codes = read_codes()

    outputfp = None
    if args.rebaseline:
        outputfp = open("results.txt", "w")
    
    suite = unittest.TestSuite()
    for method in methodlist:
        for name in names:
            if args.number != "":
                if name.split(".")[0][2:] != args.number:
                    continue
            key = (name, methods[method])
            if key in expected_codes:
                expected_code = expected_codes[key]
            else:
                expected_code = None
            suite.addTest(
                ParametrizedTestCase.parametrize(
                    QRTestCase,
                    method=method,
                    filename=name,
                    code=expected_code,
                    rebaseline=args.rebaseline,
                    outputfp=outputfp))

    unittest.TextTestRunner(verbosity=2).run(suite)
    if outputfp is not None:
        outputfp.close()
