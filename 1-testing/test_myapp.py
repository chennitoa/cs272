import unittest
import mypp_testing as mypp


class MyPPTesting(unittest.TestCase):

    def test_check_prime(self):
        self.assertFalse(mypp.check_prime(1))
        self.assertTrue(mypp.check_prime(2))

    def test_check_prime_err(self):
        self.assertRaises(TypeError, mypp.check_prime, *[1.1])
        self.assertRaises(ValueError, mypp.check_prime, *[0])

    def test_check_divide(self):
        self.assertEqual(mypp.divide(4, 2), 2)

    def test_check_divide_err(self):
        self.assertRaises(ValueError, mypp.divide, *[1, 0])

    def test_check_student(self):
        student0 = mypp.Student('Austin', 'Chen', '10000')
        self.assertEqual(student0.fullname(), 'Austin Chen')
        self.assertEqual(student0.admission_yr(), '10')

    def test_check_student_err(self):
        self.assertRaises(ValueError, lambda: mypp.Student('Austin', 'Chen', 0))


if __name__ == '__main__':
    unittest.main()
