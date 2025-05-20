import unittest
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
import os

class TestCase2(unittest.TestCase):
    username = ""
    password = ""

    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.implicitly_wait(30)
        self.base_url = "https://www.google.com/"
        self.verificationErrors = []
        self.accept_next_alert = True
        self.username = os.environ.get("username", "")
        self.password = os.environ.get("password", "")

    def test_case1(self):
        driver = self.driver
        driver.get("https://katalon-demo-cura.herokuapp.com/profile.php#login")
        driver.find_element(By.ID, "txt-username").clear()
        driver.find_element(By.ID, "txt-username").send_keys(self.username)
        driver.find_element(By.ID, "txt-password").clear()
        driver.find_element(By.ID, "txt-password").send_keys(self.password)
        driver.find_element(By.ID, "btn-login").click()
        time.sleep(1)
        try:
            message = driver.find_element(By.XPATH, "//section[@id='login']/div/div/div/p[2]").text
            self.assertEqual("Login failed! Please ensure the username and password are valid.", message)
        except AssertionError as e:
            self.verificationErrors.append(str(e))

    def tearDown(self):
        self.driver.quit()

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCase2)
    result = unittest.TextTestRunner(stream=open(os.devnull, 'w')).run(suite)
    print("pass" if result.wasSuccessful() else "fail")