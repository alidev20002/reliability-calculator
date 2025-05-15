import csv
import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import random
from faker import Faker

fake = Faker()

CSV_FILE = "automate.csv"
ROW_COUNT = 5  # how many test rows to generate

# Step 1: Generate new CSV with fake data
def generate_fake_csv():
    fieldnames = ['username', 'password']
    with open(CSV_FILE, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(ROW_COUNT):
            writer.writerow({
                'username': fake.user_name(),
                'password': fake.password(length=10, special_chars=True, digits=True, upper_case=True, lower_case=True),
            })

# Shared base class
class BaseLoginTest(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.implicitly_wait(30)
        self.base_url = "https://katalon-demo-cura.herokuapp.com/"
        self.verificationErrors = []
        self.accept_next_alert = True

    def tearDown(self):
        self.driver.quit()
        self.assertEqual([], self.verificationErrors)

# Factory function to create individual tests
def create_test(username, password):
    def test(self):
        driver = self.driver
        driver.get(self.base_url + "profile.php#login")
        driver.find_element(By.ID, "txt-username").clear()
        driver.find_element(By.ID, "txt-username").send_keys(username)
        driver.find_element(By.ID, "txt-password").clear()
        driver.find_element(By.ID, "txt-password").send_keys(password)
        driver.find_element(By.ID, "btn-login").click()
        time.sleep(2)
        try:
            message = driver.find_element(By.XPATH, "//section[@id='login']/div/div/div/p[2]").text
            self.assertEqual("Login failed! Please ensure the username and password are valid.", message)
        except AssertionError as e:
            self.verificationErrors.append(f"{username}: {str(e)}")
        driver.get(self.base_url + "logout.php")
    return test

# Load CSV and dynamically attach tests
def load_tests():
    with open(CSV_FILE, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            test_name = f"test_login_row_{i+1}_{row['username']}"
            test_func = create_test(row['username'], row['password'])
            setattr(DynamicLoginTests, test_name, test_func)

# Create a dynamic subclass
class DynamicLoginTests(BaseLoginTest):
    pass

# ==== MAIN ENTRY ====
if __name__ == "__main__":
    generate_fake_csv()  # Step 1: Refresh CSV data
    load_tests()         # Step 2: Load tests dynamically

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DynamicLoginTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n====== Test Summary ======")
    print(f"Total tests run: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("==========================")
