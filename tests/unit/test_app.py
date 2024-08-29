import unittest
from app import app
import os

class BasicTests(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home(self):
        result = self.app.get('/')
        self.assertEqual(result.status_code, 200)

    def test_upload_image_no_file(self):
        result = self.app.post('/api/upload', data={'db_url': 'mongodb://localhost:27017/test_db'})
        self.assertEqual(result.status_code, 400)
        self.assertIn(b'No file part', result.data)

    def test_upload_image_no_db_url(self):
        with open('test.jpg', 'rb') as img:
            result = self.app.post('/api/upload', data={'file': img})
            self.assertEqual(result.status_code, 400)
            self.assertIn(b'No database URL provided', result.data)


if __name__ == "__main__":
    unittest.main()
