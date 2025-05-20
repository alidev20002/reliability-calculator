import os
import json
import subprocess
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
import streamlit as st
from faker import Faker

fake = Faker()

SETTINGS_FILE = "testcases.json"


def load_all_testcases():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_all_testcases(data):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        st.rerun()


def generate_data(fields, n_rows):
    df = pd.DataFrame()
    for field in fields:
        if field["type"] == "عددی":
            if field["distribution"] == "یكنواخت":
                data = np.random.uniform(field["low"], field["high"], n_rows)
            elif field["distribution"] == "نرمال":
                data = np.random.normal(field["mean"], field["std"], n_rows)
            elif field["distribution"] == "پواسون":
                data = np.random.poisson(field["lam"], n_rows)
            if field["number_kind"] == "صحیح":
                data = data.astype(int)
            else:
                data = data.round(2)
            df[field["name"]] = data

        elif field["type"] == "متنی":
            if field["text_type"] == "نام":
                df[field["name"]] = [fake.name() for _ in range(n_rows)]
            elif field["text_type"] == "ایمیل":
                df[field["name"]] = [fake.email() for _ in range(n_rows)]
            elif field["text_type"] == "آدرس":
                df[field["name"]] = [fake.address().replace("\n", ", ") for _ in range(n_rows)]
            elif field["text_type"] == "پسورد":
                df[field["name"]] = [fake.password(length=12, special_chars=True, digits=True, upper_case=True, lower_case=True) for _ in range(n_rows)]
            else:
                df[field["name"]] = [fake.text(max_nb_chars=50) for _ in range(n_rows)]

        elif field["type"] == "چند گزینه‌ای":
            opts = field["options"]
            if field.get("prob"):
                probs = [float(p) for p in field["prob"].split(",")]
                df[field["name"]] = np.random.choice(opts, size=n_rows, p=probs)
            else:
                df[field["name"]] = np.random.choice(opts, size=n_rows)

        elif field["type"] == "تاریخ":
            start = datetime.combine(field["start_date"], datetime.min.time())
            end = datetime.combine(field["end_date"], datetime.min.time())
            df[field["name"]] = [start + timedelta(days=np.random.randint(0, (end - start).days + 1))
                                 for _ in range(n_rows)]
    return df

def start_tester_test(tests, settings):
    number_of_tests_before_first_failure = 0

    for test in tests:
        n_rows = settings['total_number_of_tests'] * (tests[test]['percent'] / 100.0)
        df = generate_data(tests[test]["fields"], math.ceil(n_rows))
        df['result'] = ''
        csv_name = tests[test]["testcase_name"].replace('.py', '') + '-tester' + str(i+1) + '.csv'
        csv_path = os.path.join(tests[test]["testcase_dir"], csv_name)
        df.to_csv(csv_path, index=False)

        for idx, row in df.iterrows():
            env = os.environ.copy()
            env.update(row.dropna().astype(str).to_dict())
            try:
                result = subprocess.run(
                    ["python", os.path.join(tests[test]["testcase_dir"], tests[test]["testcase_name"])],
                    env=env,
                    capture_output=True,
                    text=True, timeout=300
                )
                output = result.stdout.strip().splitlines()
                outcome = next((line.strip() for line in output if line.strip() in ("pass", "fail")), "fail")

                df.at[idx, "result"] = outcome
                df.to_csv(csv_path, index=False)

                number_of_tests_before_first_failure += 1

                if outcome == 'fail':
                    return number_of_tests_before_first_failure, True
            except Exception as e:
                return "", str(e)
    return number_of_tests_before_first_failure, False

def run_testcase(tests, settings):
    number_of_testers = settings['number_of_testers']
    number_of_failures = 0
    total_number_of_tests_executed = 0
    for i in range(number_of_testers):
        number_of_tests, isFailed = start_tester_test(tests, settings)
        total_number_of_tests_executed += number_of_tests
        if isFailed:
            number_of_failures += 1
        

st.set_page_config(page_title="سیستم مدیریت و اجرای آزمون", layout="wide")
st.markdown("<h1 style='text-align: center;'>🧪 سیستم مدیریت و اجرای آزمون</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["مدیریت آزمون‌ها و تولید داده", "اجرای آزمون‌ها"])


with tab1:
    # Load all testcases
    all_testcases = load_all_testcases()

    st.sidebar.title("🗂️ مدیریت سناریوهای آزمون")
    test_names = list(all_testcases.keys())
    selected_test = st.sidebar.selectbox("🔽 انتخاب یا ساخت سناریو آزمون", ["سناریو جدید"] + test_names)

    if selected_test == "سناریو جدید":
        new_test_name = st.sidebar.text_input("📝 نام سناریو جدید")
        test_data = {}
    else:
        new_test_name = selected_test
        test_data = all_testcases[selected_test]

    # Main Inputs
    testcase_dir = st.text_input("📂 مسیر دایرکتوری سناریو آزمون", value=test_data.get("testcase_dir", ""))
    testcase_name = st.text_input("📄 نام فایل اسکریپت آزمون", value=test_data.get("testcase_name", ""))
    percent = st.number_input("ضریب اهمیت (درصد)", min_value=0, max_value=100, step=1)

    # Define fields
    fields = []
    n_fields = st.number_input("تعداد فیلدها", min_value=1, value=1)

    for i in range(n_fields):
        st.markdown(f"### ⚙️ تنظیمات فیلد {i+1}")
        col1, col2 = st.columns(2)
        with col1:
            field_name = st.text_input(f"نام فیلد {i+1}", key=f"name_{i}")
        with col2:
            field_type = st.selectbox(f"نوع فیلد {i+1}", ["عددی", "متنی", "چند گزینه‌ای", "تاریخ"], key=f"type_{i}")

        config = {"name": field_name, "type": field_type}

        if field_type == "عددی":
            number_kind = st.selectbox("نوع عدد", ["صحیح", "اعشاری"], key=f"num_kind_{i}")
            config["number_kind"] = number_kind

            dist = st.selectbox(f"توزیع برای فیلد {field_name}", ["یكنواخت", "نرمال", "پواسون"], key=f"dist_{i}")
            config["distribution"] = dist

            if dist == "یكنواخت":
                config["low"] = st.number_input("حداقل مقدار", key=f"low_{i}", value=0)
                config["high"] = st.number_input("حداکثر مقدار", key=f"high_{i}", value=100)
            elif dist == "نرمال":
                config["mean"] = st.number_input("میانگین", key=f"mean_{i}", value=0)
                config["std"] = st.number_input("انحراف معیار", key=f"std_{i}", value=1)
            elif dist == "پواسون":
                config["lam"] = st.number_input("لامبدا (میانگین)", key=f"lam_{i}", value=5)

        elif field_type == "متنی":
            config["text_type"] = st.selectbox("نوع متن", ["نام", "پسورد", "ایمیل", "آدرس", "جمله تصادفی"], key=f"text_{i}")

        elif field_type == "چند گزینه‌ای":
            options = st.text_input("گزینه‌ها را با کاما جدا کنید (مثلاً: پایین,متوسط,بالا)", key=f"options_{i}")
            config["options"] = [opt.strip() for opt in options.split(",") if opt.strip()]
            config["prob"] = st.text_input("احتمال گزینه‌ها (مثلاً: 0.2,0.5,0.3) یا خالی برای برابر", key=f"prob_{i}")

        elif field_type == "تاریخ":
            config["start_date"] = st.date_input("تاریخ شروع", key=f"start_{i}")
            config["end_date"] = st.date_input("تاریخ پایان", key=f"end_{i}")

        fields.append(config)

    # Save test case
    if st.button("💾 ذخیره این سناریو آزمون"):
        if not new_test_name:
            st.error("⚠️ لطفاً نام سناریو آزمون را وارد کنید.")
        else:
            all_testcases[new_test_name] = {
                "testcase_dir": testcase_dir,
                "testcase_name": testcase_name,
                "percent": percent,
                "fields": fields
            }
            save_all_testcases(all_testcases)
            st.success(f"✅ سناریو '{new_test_name}' ذخیره شد.")

    # Show list of test cases
    st.sidebar.markdown("### 📋 لیست سناریوها")
    st.sidebar.markdown("برای حذف هر سناریو بر روی آن کلیک کنید")
    for name in all_testcases:
         if st.sidebar.button(f"- **{name}**   ---  {all_testcases[name]['percent']}"):
             all_testcases.pop(name, None)
             save_all_testcases(all_testcases)


with tab2:
    input_rate = st.number_input("تعداد درخواست‌ها به سیستم در یک ساعت", min_value=1, max_value=1000, step=1)
    number_of_testers = st.number_input("تعداد آزمونگرهای سیستم", min_value=1, max_value=10, step=1)
    test_duration = st.number_input("تعداد ساعات تست", min_value=1, max_value=24)
    total_number_of_tests = test_duration * input_rate
    settings = {
        'total_number_of_tests': total_number_of_tests,
        'number_of_testers': number_of_testers,
        'input_rate': input_rate
    }
    if st.button("اجرای آزمون‌ها"):
        st.info("🛠️ در حال اجرای آزمون‌ها و تولید داده...")
        run_testcase(all_testcases, settings)
        st.success("✅ آزمون‌ها با موفقیت اجرا شدند.")