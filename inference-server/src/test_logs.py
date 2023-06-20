import unittest

from logger import LOGGER


class MyTestCase(unittest.TestCase):
    def test_something(self):
        LOGGER.info('test')
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
