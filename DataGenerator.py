import os
import json
import subprocess
from datetime import datetime, timedelta

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


def run_testcase(settings, n_rows):
    df = generate_data(settings["fields"], n_rows)
    csv_path = os.path.join(settings["testcase_dir"], settings["csv_name"])
    df.to_csv(csv_path, index=False)
    try:
        result = subprocess.run(
            ["python", os.path.join(settings["testcase_dir"], settings["testcase_name"])],
            capture_output=True, text=True, timeout=300
        )
        return result.stdout, result.stderr
    except Exception as e:
        return "", str(e)


# Load all testcases
all_testcases = load_all_testcases()

st.title("🧪 سیستم مدیریت و اجرای تست‌کیس")

# Sidebar: Select or create test case
st.sidebar.title("🗂️ مدیریت تست‌کیس‌ها")
test_names = list(all_testcases.keys())
selected_test = st.sidebar.selectbox("🔽 انتخاب یا ساخت تست‌کیس", ["<جدید>"] + test_names)

if selected_test == "<جدید>":
    new_test_name = st.sidebar.text_input("📝 نام تست‌کیس جدید")
    test_data = {}
else:
    new_test_name = selected_test
    test_data = all_testcases[selected_test]

# Main Inputs
testcase_dir = st.text_input("📂 مسیر دایرکتوری TestCase", value=test_data.get("testcase_dir", ""))
testcase_name = st.text_input("📄 نام فایل TestCase", value=test_data.get("testcase_name", ""))
file_name = st.text_input("🧾 نام فایل CSV", value=test_data.get("csv_name", "test_data.csv"))
interval_seconds = st.number_input("⏲️ دوره تناوب اجرای تست (ثانیه)", min_value=1, max_value=86400,
                                   value=test_data.get("interval", 60))
n_rows = st.number_input("📊 تعداد ردیف داده", min_value=1, value=test_data.get("n_rows", 10))

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
if st.button("💾 ذخیره این تست‌کیس"):
    if not new_test_name:
        st.error("⚠️ لطفاً نام تست‌کیس را وارد کنید.")
    else:
        all_testcases[new_test_name] = {
            "testcase_dir": testcase_dir,
            "testcase_name": testcase_name,
            "csv_name": file_name,
            "interval": interval_seconds,
            "n_rows": n_rows,
            "fields": fields
        }
        save_all_testcases(all_testcases)
        st.success(f"✅ تست‌کیس '{new_test_name}' ذخیره شد.")

# Show list of test cases
st.sidebar.markdown("### 📋 لیست تست‌کیس‌ها")
for name in all_testcases:
    st.sidebar.markdown(f"- **{name}**: {all_testcases[name]['csv_name']}")

# Run selected test case
if selected_test != "<جدید>":
    if st.button("🚀 اجرای تست‌کیس"):
        settings = all_testcases[selected_test]
        st.info("🛠️ در حال اجرای تست‌کیس و تولید داده...")
        stdout, stderr = run_testcase(settings, settings["n_rows"])
        if stderr:
            st.error(f"❌ خطا در اجرای تست:\n```\n{stderr}\n```")
        else:
            st.success("✅ تست با موفقیت اجرا شد.")
            st.text(f"📤 خروجی:\n{stdout}")
