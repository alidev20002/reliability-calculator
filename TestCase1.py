import unittest
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
import os

class TestCase1(unittest.TestCase):
    username = ""
    password = ""

    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.implicitly_wait(30)
        self.base_url = "https://www.google.com/"
        self.verificationErrors = []
        self.accept_next_alert = True
        self.username = os.environ.get("USERNAME", "")
        self.password = os.environ.get("PASSWORD", "")

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

# Generate suite from CSV
def load_tests():
    suite = unittest.TestSuite()

    # with open(CSV_FILE, newline='') as f:
    #     reader = csv.DictReader(f)
    #     for i, row in enumerate(reader):
    #         username = row.get("username", "")
    #         password = row.get("password", "")
    #         test_name = f"test_case1_row_{i+1}_{username}"

    #         def test_template(self, u=username, p=password):
    #             self.username = u
    #             self.password = p
    #             self.test_case1()

    #         setattr(TestCase1, test_name, test_template)
    #         suite.addTest(TestCase1(test_name))
    # return suite


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCase1)
    result = unittest.TextTestRunner(stream=open(os.devnull, 'w')).run(suite)
    print("pass" if result.wasSuccessful() else "fail")
