import unittest
from app import app, get_db_connection
from io import BytesIO
import os

class IntegrationTests(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.db_url = 'mongodb://localhost:27017/test_db'
        self.db = get_db_connection(self.db_url)
        self.collection = self.db['image_features']

    def tearDown(self):
        self.db.client.drop_database('test_db')

    def test_upload_and_search(self):
        with open('test.jpg', 'rb') as img:
            result = self.app.post('/api/upload', data={'file': img, 'db_url': self.db_url})
            self.assertEqual(result.status_code, 200)
            response_data = result.get_json()
            self.assertTrue(response_data['success'])

            file_id = response_data['images'][0]
            search_result = self.app.get(f'/api/get_similar_images?image={file_id}&db_url={self.db_url}')
            self.assertEqual(search_result.status_code, 200)
            search_data = search_result.get_json()
            self.assertTrue(search_data['success'])

    def test_upload_zip(self):
        zip_file_path = os.path.join(os.path.dirname(__file__), 'train.zip')
        with open(zip_file_path, 'rb') as zip_file:
            zip_content = zip_file.read()

    
        data = {
            'file': (BytesIO(zip_content), 'train.zip',  'application/zip'),
            'db_url': self.db_url
        }

        result = self.app.post('/api/upload_zip', data=data, content_type='multipart/form-data')

        if result.status_code != 200:
            print("Request Form Data: ", data)

        self.assertEqual(result.status_code, 200)
        response_data = result.get_json()
        self.assertTrue(response_data['success'])

    def test_get_image(self):
        with open('test.jpg', 'rb') as img:
            upload_result = self.app.post('/api/upload', data={'file': img, 'db_url': self.db_url})
            self.assertEqual(upload_result.status_code, 200)
            upload_data = upload_result.get_json()
            self.assertTrue(upload_data['success'])
            image_id = upload_data['images'][0]

        get_result = self.app.get(f'/api/get_image/{image_id}?db_url={self.db_url}')
        self.assertEqual(get_result.status_code, 200)
        
        self.assertIn(get_result.content_type, ['image/png', 'application/octet-stream'])


if __name__ == "__main__":
    unittest.main()
