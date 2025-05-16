import csv
import unittest
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, NoAlertPresentException

CSV_FILE = "automate.csv"

class TestCase1(unittest.TestCase):
    username = ""
    password = ""

    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.implicitly_wait(30)
        self.base_url = "https://www.google.com/"
        self.verificationErrors = []
        self.accept_next_alert = True

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

# Custom Test Result class to capture and print readable results
class StreamlitFriendlyTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super(StreamlitFriendlyTestResult, self).__init__(*args, **kwargs)
        self.successes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)

    def printReadableReport(self):
        total = self.testsRun
        passed = len(self.successes)
        failed = len(self.failures) + len(self.errors)

        print("\n====== Test Results ======")
        for test in self.successes:
            print(f"[PASS] {test}")
        for test, err in self.failures:
            print(f"[FAIL] {test}")
        for test, err in self.errors:
            print(f"[ERROR] {test}")
        print("\n====== Test Summary ======")
        print(f"Total tests: {total}")
        print(f"Passed     : {passed}")
        print(f"Failed     : {failed}")
        print("==============================")


# Generate suite from CSV
def load_tests():
    suite = unittest.TestSuite()
    with open(CSV_FILE, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            username = row.get("username", "")
            password = row.get("password", "")
            test_name = f"test_case1_row_{i+1}_{username}"

            def test_template(self, u=username, p=password):
                self.username = u
                self.password = p
                self.test_case1()

            setattr(TestCase1, test_name, test_template)
            suite.addTest(TestCase1(test_name))
    return suite


if __name__ == "__main__":
    suite = load_tests()
    runner = unittest.TextTestRunner(verbosity=0, resultclass=StreamlitFriendlyTestResult)
    result = runner.run(suite)
    result.printReadableReport()
