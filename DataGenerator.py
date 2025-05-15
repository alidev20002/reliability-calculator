import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()

st.title("🎲 ابزار تولید داده تست با توزیع های مختلف")

# تعداد ردیف و نام فایل
n_rows = st.number_input("تعداد ردیف‌های داده (نمونه آزمون)", min_value=1, value=10)
file_name = st.text_input("نام فایل خروجی (CSV)", value="test_data.csv")

# ذخیره فیلدها
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
        config["text_type"] = st.selectbox("نوع متن", ["نام", "ایمیل", "آدرس", "جمله تصادفی"], key=f"text_{i}")

    elif field_type == "چند گزینه‌ای":
        options = st.text_input("گزینه‌ها را با کاما جدا کنید (مثلاً: پایین,متوسط,بالا)", key=f"options_{i}")
        config["options"] = [opt.strip() for opt in options.split(",") if opt.strip()]
        config["prob"] = st.text_input("احتمال گزینه‌ها (مثلاً: 0.2,0.5,0.3) یا خالی برای برابر", key=f"prob_{i}")

    elif field_type == "تاریخ":
        config["start_date"] = st.date_input("تاریخ شروع", key=f"start_{i}")
        config["end_date"] = st.date_input("تاریخ پایان", key=f"end_{i}")

    fields.append(config)

if st.button("🚀 تولید فایل CSV"):
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
            else:
                df[field["name"]] = [fake.text(max_nb_chars=50) for _ in range(n_rows)]

        elif field["type"] == "چند گزینه‌ای":
            opts = field["options"]
            if field["prob"]:
                probs = [float(p) for p in field["prob"].split(",")]
                df[field["name"]] = np.random.choice(opts, size=n_rows, p=probs)
            else:
                df[field["name"]] = np.random.choice(opts, size=n_rows)

        elif field["type"] == "تاریخ":
            start = datetime.combine(field["start_date"], datetime.min.time())
            end = datetime.combine(field["end_date"], datetime.min.time())
            df[field["name"]] = [start + timedelta(days=random.randint(0, (end - start).days)) for _ in range(n_rows)]

    # ذخیره فایل CSV
    df.to_csv(file_name, index=False)
    st.success(f"✅ فایل CSV با موفقیت ساخته شد: {file_name}")
    st.download_button("📥 دانلود فایل CSV", data=df.to_csv(index=False).encode('utf-8'), file_name=file_name, mime='text/csv')
