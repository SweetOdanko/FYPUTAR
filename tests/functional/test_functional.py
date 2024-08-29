from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import unittest
import os
import time

class FunctionalTests(unittest.TestCase):

    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.get("http://localhost:5000") 

    def tearDown(self):
        self.driver.quit()

    def test_home_page(self):
        driver = self.driver
        self.assertIn("ImageQuest", driver.title)

    def test_upload_image_and_download(self):
        driver = self.driver
        wait = WebDriverWait(driver, 20) 

        file_input = wait.until(EC.presence_of_element_located((By.ID, "fileInput")))
        file_input.send_keys("C:/Users/Wai Soon/Desktop/Intelligent Image Search Engine with AI-Based Similarity Detection/test.jpg")

        wait.until(EC.presence_of_element_located((By.ID, "similarImages")))

        similar_images = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#similarImages img")))

        similar_image = similar_images[0]
        similar_image.click()

        wait.until(EC.presence_of_element_located((By.ID, "downloadButton")))

        download_button = driver.find_element(By.ID, "downloadButton")
        download_button.click()

        time.sleep(5)


if __name__ == "__main__":
    unittest.main()
