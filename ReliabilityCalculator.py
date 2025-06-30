import json
import os
from flet import *
import time
import pandas as pd
import subprocess
import math
import threading
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit, root
from scipy.stats import poisson
import requests

SETTINGS_FILE = 'testcases.json'
RESULTS_FILE = 'growth/results.json'
LLM_GENERATE_DATA_API_URL = 'http://localhost:5000/generate_test_cases'

os.makedirs('growth', exist_ok=True)
os.makedirs('test_and_estimate', exist_ok=True)

def load_all_testcases():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_all_testcases(data):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_results(data):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def generate_input_data(test_case_name, test_case_path, tester_id, count, is_growth):
    dir_name, _ = os.path.split(test_case_path)
    if is_growth:
        model_dir = 'growth'
    else:
        model_dir = 'test_and_estimate'

    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filename = f"{test_case_name}-tester{tester_id}.csv"
    csv_path = os.path.join(model_dir, csv_filename)
    csv_path = os.path.join(current_dir, csv_path)

    html_path = os.path.join(dir_name, 'katalon_test.html')
    print(html_path)
    request_body = {
        'katalon_path': html_path,
        'output_csv_path': csv_path,
        'num_test_cases': count
    }

    try:
        response = requests.post(url=LLM_GENERATE_DATA_API_URL, json=request_body)
        if response.status_code == 200:
            print("POST request successful!")
            return csv_path
        else:
            print(f"POST request failed with status code: {response.status_code}")
            print("Response content:", response.text)
    except:
        pass

    return None

def plot_failure_rate_change(data):
    x = [item['failure_rate'] for item in data]
    y = [item['cumulative_failures'] for item in data]

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel("Failure Rate")
    ax.set_ylabel("Cumulative Failures")
    ax.set_title("Failure Rate vs Cumulative Failures")

    os.makedirs('growth/plots', exist_ok=True)
    filename = f"growth/plots/plot_failure_rate_change.png"
    plt.savefig(filename)
    plt.close(fig)

    return filename

def plot_failure_detection_rate(data):
    x = [item['cumulative_time'] for item in data]
    y = [item['failures'] for item in data]

    fig, ax = plt.subplots()
    ax.bar(x, y, width=5.0, edgecolor='black')  # Adjust width as needed
    ax.set_xlabel("Cumulative Time")
    ax.set_ylabel("Number of Failures")
    ax.set_title("Number of Failures vs Cumulative Time")

    os.makedirs("growth/plots", exist_ok=True)
    filename = "growth/plots/plot_failure_detection_rate.png"
    plt.savefig(filename)
    plt.close(fig)

    return filename

def estimate_goel_okumoto(data, t):
    X = np.array([[item["failure_rate"]] for item in data])
    y = np.array([item["cumulative_failures"] for item in data])

    model = LinearRegression()
    model.fit(X, y)

    a = model.intercept_
    b = abs(1.0 / model.coef_[0])
    total_time = data[-1]['cumulative_time']

    f = a * b * math.exp(-b * total_time)

    return math.exp(-f * t)

def estimate_weibull(data, t):
    # ti: time intervals, ki: number of faults observed
    # time intervals t_i
    ti = np.array([item["cumulative_time"] for item in data]) 
    # corresponding fault counts k_i  
    ki = np.array([item["failures"] for item in data])   

    n = len(ti)
    ti_1 = np.roll(ti, 1)
    ti_1[0] = 0  # t_0 = 0

    def equations(vars):
        b, c = vars
        term1 = 0
        term2 = 0
        sum_ki = np.sum(ki)

        for i in range(n):
            num1 = ki[i] * ((ti[i] ** c) * np.exp(-b * ti[i] ** c) - (ti_1[i] ** c) * np.exp(-b * ti_1[i] ** c))
            den1 = np.exp(-b * ti_1[i] ** c) - np.exp(-b * ti[i] ** c)
            term1 += num1 / den1

            num2 = ki[i] * (
                (ti[i] ** c) * np.log(ti[i]) * np.exp(-b * ti[i] ** c)
                - (ti_1[i] ** c) * np.log(ti_1[i] + 1e-10) * np.exp(-b * ti_1[i] ** c)
            )
            den2 = np.exp(-b * ti_1[i] ** c) - np.exp(-b * ti[i] ** c)
            term2 += num2 / den2

        eq1 = term1 - (ti[n-1] ** c) * np.exp(-b * ti[n-1] ** c) * sum_ki / (1 - np.exp(-b * ti[n-1] ** c))
        eq2 = term2 - b * (ti[n-1] ** c) * np.log(ti[n-1]) * np.exp(-b * ti[n-1] ** c) * sum_ki / (1 - np.exp(-b * ti[n-1] ** c))

        return [eq1, eq2]

    # Initial guesses for b and c
    initial_guess = [0.01, 1.0]
    sol = root(equations, initial_guess)

    if sol.success:
        b, c = sol.x
        denom = 1 - np.exp(-b * ti[-1] ** c)
        a = np.sum(ki) / denom
        print(f"تخمین پارامترها:\na = {a:.4f}, b = {b:.4f}, c = {c:.4f}")
    else:
        print("حل معادلات همگرا نشد.")
        return 0, 'در حال حاضر استفاده از این مدل توصیه نمی‌شود'
    
    total_time = data[-1]['cumulative_time']
    
    f = (a * b * c) * ((b * total_time) ** (c-1)) * math.exp(-((b * total_time) ** c))
    return math.exp(-f * t), None

def estimate_log_logistics(data, t):
    
    def F_model(t, a, b, c):
        return (a * (b * t)**c) / (1 + (b * t)**c)
    
    initial_guess = [35, 0.1, 2]
    cumulative_time = np.array([item["cumulative_time"] for item in data])
    cumulative_failures = np.array([item["cumulative_failures"] for item in data])
    params, _ = curve_fit(F_model, cumulative_time, cumulative_failures, p0=initial_guess, bounds=(0, np.inf))

    a, b, c = params
    total_time = data[-1]['cumulative_time']
    f = (a * b * c * ((b * total_time) ** (c - 1))) / ((1 + ((b * total_time) ** c)) ** 2)

    return math.exp(-f * t)

def estimate_duane(data, t):
    cumulative_time = np.array([item["cumulative_time"] for item in data])
    cumulative_failures = np.array([item["cumulative_failures"] for item in data])

    ln_t = np.log(cumulative_time).reshape(-1, 1)
    ln_f = np.log(cumulative_failures)

    model = LinearRegression()
    model.fit(ln_t, ln_f)

    b = model.coef_[0]
    ln_a = model.intercept_
    a = np.exp(ln_a)
    total_time = data[-1]['cumulative_time']
    f = a * b * (total_time ** (b - 1))
    
    return math.exp(-f * t)

def test_and_estimation_reliability(total_failures, total_time, t):
    failure_rate = float(total_failures) / total_time
    return math.exp(-failure_rate * t)

def build_tab_manage_tests(page: Page):
    all_testcases = load_all_testcases()

    selected_test = Dropdown(
        label="انتخاب یا ساخت سناریو آزمون",
        text_align='right',
        text_style=TextStyle(
            size=14
        ),
        expand=True
    )
    testcase_dir_picker = FilePicker()
    page.overlay.append(testcase_dir_picker)
    testcase_dir_input = TextField(
        label="📂 مسیر دایرکتوری سناریو آزمون",
        read_only=True
    )

    testcase_dir_row = Row(
        controls=[
            testcase_dir_input,
            ElevatedButton(
                text="انتخاب فایل",
                icon=Icons.UPLOAD_FILE,
                bgcolor=Colors.BLUE_500,
                color=Colors.WHITE,
                style=ButtonStyle(
                    shape= RoundedRectangleBorder(8),
                    padding=Padding(15, 15, 15, 15)
                ),
                on_click=lambda e: testcase_dir_picker.pick_files(allow_multiple=False)
            )
        ],
        spacing=10
    )

    def on_file_selected(e):
        if e.files:
            testcase_dir_input.value = e.files[0].path
            testcase_dir_input.update()

    testcase_dir_picker.on_result = on_file_selected

    max_percent_value = 100 - sum(item['percent'] for item in all_testcases.values())
    percent_input = Slider(min=0, max=max_percent_value, divisions=max_percent_value, label="{value}%", value=0)
    new_test_name_input = TextField(label="نام سناریو جدید")
    test_list_column = Column(scroll=ScrollMode.AUTO, height=300)
    user_message = Text()

    def refresh_test_list():
        selected_test.options = [dropdown.Option("سناریو جدید")] + [dropdown.Option(k) for k in all_testcases]
        test_list_column.controls = []
        for name, info in all_testcases.items():
            test_list_column.controls.append(
                ListTile(
                    title=Text(f"{name}"),
                    subtitle=Text(f"ضریب اهمیت: {info['percent']}%", rtl=True),
                    trailing=Icon(name="DELETE", color="red"),
                    on_click=lambda e, name=name: delete_testcase(name)
                )
            )
        page.update()

    def delete_testcase(name):
        all_testcases.pop(name, None)
        save_all_testcases(all_testcases)
        user_message.value = f"سناریو '{name}' حذف شد."
        refresh_test_list()

    def save_testcase(e):
        new_name = new_test_name_input.value
        if not new_name:
            user_message.value = "لطفاً نام سناریو آزمون را وارد کنید."
            user_message.color = Colors.RED
            page.update()
            return
        
        if not testcase_dir_input.value:
            user_message.value = "لطفاً مسیر دایرکتوری سناریو آزمون را مشخص کنید."
            user_message.color = Colors.RED
            page.update()
            return
        
        if percent_input.value == 0:
            user_message.value = "ضریب اهمیت سناریوی آزمون نمی‌تواند صفر باشد."
            user_message.color = Colors.RED
            page.update()
            return
        
        all_testcases[new_name] = {
            "testcase_dir": testcase_dir_input.value,
            "percent": int(percent_input.value),
        }
        save_all_testcases(all_testcases)
        percent_input.value = 0
        max_percent_value = 100 - sum(item['percent'] for item in all_testcases.values())
        percent_input.max = max_percent_value
        percent_input.divisions = max_percent_value
        user_message.value = f"سناریو '{new_name}' ذخیره شد."
        user_message.color = Colors.GREEN
        refresh_test_list()

    def on_test_select(e):
        name = selected_test.value
        if name and name in all_testcases:
            data = all_testcases[name]
            new_test_name_input.value = name
            new_test_name_input.read_only = True
            testcase_dir_input.value = data["testcase_dir"]
            percent_input.max = 100
            percent_input.divisions = 100
            percent_input.value = data["percent"]
            percent_input.disabled = True
        else:
            new_test_name_input.value = ""
            new_test_name_input.read_only = False
            testcase_dir_input.value = ""
            max_percent_value = 100 - sum(item['percent'] for item in all_testcases.values())
            percent_input.max = max_percent_value
            percent_input.divisions = max_percent_value
            percent_input.value = 0
            percent_input.disabled = False
        page.update()

    selected_test.on_change = on_test_select

    refresh_test_list()

    return Column([
        Container(
            content=Text("مدیریت سناریوهای آزمون", size=20, weight="bold", text_align="center"),
            alignment=alignment.center,
            padding=30
        ),
        Row([
            Container(
                Column([
                    selected_test,
                    Column([
                        Text("لیست سناریوها", style=TextThemeStyle.TITLE_MEDIUM),
                        test_list_column,
                    ])
                ], width=300, horizontal_alignment="center", alignment="start", spacing=50),
                bgcolor="#dfdfdf",
                padding=30
            ),
            
            Column([
                new_test_name_input,
                testcase_dir_row,
                Row([
                    Text('ضریب اهمیت سناریوی آزمون (درصد): '),
                    percent_input,
                ]),
                ElevatedButton(
                    text="ذخیره سناریو",
                    bgcolor=Colors.BLUE_500,
                    color=Colors.WHITE,
                    style=ButtonStyle(
                        shape= RoundedRectangleBorder(8),
                        padding=Padding(15, 15, 15, 15)
                    ),
                    on_click=save_testcase
                ),
                user_message
            ], horizontal_alignment="start")
        ], alignment='start', vertical_alignment='start', expand=True, spacing=50),
    ], spacing=50)

def build_tab_growth_model_run_tests(page: Page):
    all_testcases = load_all_testcases()

    number_of_failures = 0
    total_execution_time = 0
    running_threads = []
    thread_statuses = Column(width=600, spacing=10, scroll=ScrollMode.AUTO)

    def start_tester_test(testerId):
        nonlocal number_of_failures, total_execution_time, running_threads
        start_time = time.time()

        for test_case in all_testcases:
            number_of_sub_tests = math.ceil((int(number_of_tests.value) * all_testcases[test_case]['percent']) / 100)

            thread_statuses.controls[testerId].subtitle = Text(f"در حال ساخت داده آزمون برای سناریوی {test_case}")
            page.update()

            test_case_path = all_testcases[test_case]['testcase_dir']
            csv_path = generate_input_data(test_case, test_case_path, str(testerId + 1), number_of_sub_tests, True)
            if not csv_path:
                thread_statuses.controls[testerId].subtitle = Text(f"خطا در ساخت داده ورودی!")
                thread_statuses.controls[testerId].trailing = Icon(Icons.ERROR, color='red')
                page.update()

                running_threads[testerId] = False
                return

            df = pd.read_csv(csv_path)
            df['result'] = ''

            for idx, row in df.iterrows():
                thread_statuses.controls[testerId].subtitle = Text(f"سناریوی {test_case}: در حال اجرای آزمون {idx+1}ام از {number_of_sub_tests} آزمون")
                page.update()

                env = os.environ.copy()
                env.update(row.dropna().astype(str).to_dict())
                try:
                    result = subprocess.run(
                        ["python", test_case_path],
                        env=env,
                        capture_output=True,
                        text=True, timeout=300
                    )
                    output = result.stdout.strip().splitlines()
                    outcome = next((line.strip() for line in output if line.strip() in ("pass", "fail")), "fail")

                    df.at[idx, "result"] = outcome
                    df.to_csv(csv_path, index=False)

                    if outcome == 'fail':
                        elapsed = int(time.time() - start_time)
                        elapsed_formatted = f"{elapsed // 60:02}:{elapsed % 60:02}"

                        total_execution_time += elapsed
                        number_of_failures += 1
                        running_threads[testerId] = False

                        thread_statuses.controls[testerId].subtitle = Text(f"در آزمون {idx+1}ام از سناریوی {test_case} شکست خورد\nزمان سپری شده: {elapsed_formatted}")
                        thread_statuses.controls[testerId].trailing = Icon(Icons.ERROR, color='red')
                        page.update()
                        return
                except Exception as e:
                    
                    print(e)

                    elapsed = int(time.time() - start_time)
                    elapsed_formatted = f"{elapsed // 60:02}:{elapsed % 60:02}"

                    total_execution_time += elapsed
                    number_of_failures += 1
                    running_threads[testerId] = False

                    thread_statuses.controls[testerId].subtitle = Text(f"در آزمون {idx+1}ام از سناریوی {test_case} شکست خورد\nزمان سپری شده: {elapsed_formatted}")
                    thread_statuses.controls[testerId].trailing = Icon(Icons.ERROR, color='red')
                    page.update()
                    return
            
        elapsed = int(time.time() - start_time)
        elapsed_formatted = f"{elapsed // 60:02}:{elapsed % 60:02}"

        total_execution_time += elapsed
        running_threads[testerId] = False

        thread_statuses.controls[testerId].subtitle = Text(f"همه آزمون‌ها بدون خطا پاس شدند\nزمان سپری شده: {elapsed_formatted}")
        thread_statuses.controls[testerId].trailing = Icon(name=Icons.DONE, color='Green')
        page.update()

    def run_testcase(e):
        nonlocal running_threads, number_of_failures, total_execution_time
        total_execution_time = 0
        number_of_failures = 0
        running_threads = [False] * int(number_of_testers.value)
        thread_statuses.controls.clear()
        start_tests_button.disabled = True

        for i in range(int(number_of_testers.value)):
            thread_statuses.controls.append(
                ListTile(
                    title=Text(f"آزمونگر {i+1}"),
                    subtitle=Text("در حال آماده سازی..."),
                    trailing=ProgressRing(width=15, height=15),
                    bgcolor="#dfdfdf"
                )
            )
            t = threading.Thread(target=start_tester_test, args=(i,), daemon=True)
            t.start()
            running_threads[i] = True
        
        page.update()

        while (True in running_threads):
            pass

        start_tests_button.disabled = False

        total_time_formatted = f"{total_execution_time // 60:02}:{total_execution_time % 60:02}"
        thread_statuses.controls.append(
            ListTile(
                title=Text("پایان فرایند آزمون"),
                subtitle=Text(f"تعداد کل خرابی‌ها: {number_of_failures}\nزمان سپری شده: {total_time_formatted}"),
                bgcolor="#84c5fd"
            )
        )
        page.update()

        results = load_results()
        last_result = results[-1] if results else {"cumulative_failures": 0, "cumulative_time": 0}
        cumulative_failures = last_result["cumulative_failures"] + number_of_failures
        current_time = total_execution_time
        cumulative_time = last_result["cumulative_time"] + current_time
        results.append({
            "failures": number_of_failures,
            "time": current_time,
            "cumulative_failures": cumulative_failures,
            "cumulative_time": cumulative_time,
            "failure_rate": cumulative_failures / cumulative_time
        })
        save_results(results)

    number_of_tests = TextField(label="تعداد کل تست‌ها", value="10", keyboard_type=KeyboardType.NUMBER)
    number_of_testers = TextField(label="تعداد آزمونگرها", value="1", keyboard_type=KeyboardType.NUMBER)

    start_tests_button = ElevatedButton(
        text="اجرای آزمون‌ها",
        bgcolor=Colors.BLUE_500,
        color=Colors.WHITE,
        style=ButtonStyle(
            shape= RoundedRectangleBorder(8),
            padding=Padding(15, 15, 15, 15)
        ),
        on_click=run_testcase
    )

    return Column([
        Container(
            content=Text("اجرای آزمون‌ها", style=TextThemeStyle.HEADLINE_MEDIUM),
            alignment=alignment.center,
            padding=30
        ),
        Column([
            number_of_tests,
            number_of_testers,
            start_tests_button
        ], width=300, horizontal_alignment='center'),
        thread_statuses
    ], expand=True, horizontal_alignment='center')

def build_tab_growth_reliability(page: Page):
    results = load_results()

    rows = [
        DataRow(
            cells=[
                DataCell(Text(str(idx + 1))),
                DataCell(Text(str(item["failures"]))),
                DataCell(Text(f"{item["time"] // 60:02}:{item["time"] % 60:02}")),
                DataCell(Text(str(item["cumulative_failures"]))),
                DataCell(Text(f"{item["cumulative_time"] // 60:02}:{item["cumulative_time"] % 60:02}")),
                DataCell(Text(f"{item['failure_rate']:.4f}"))
            ]
        )
        for idx, item in enumerate(results)
    ]

    table = DataTable(
        columns=[
            DataColumn(label=Text("ردیف")),
            DataColumn(label=Text("تعداد شکست‌ها")),
            DataColumn(label=Text("زمان")),
            DataColumn(label=Text("شکست‌های تجمعی")),
            DataColumn(label=Text("زمان تجمعی")),
            DataColumn(label=Text("نرخ شکست")),
        ],
        rows=rows
    )

    image_path = plot_failure_rate_change(results)
    image_control = Image(src=image_path, width=400, height=300)

    selected_plot = Dropdown(
        label="انتخاب نمودار",
        options=[dropdown.Option('نمودار تغییر نرخ خرابی'), dropdown.Option('نمودار نرخ کشف خرابی')],
        text_align='right',
        text_style=TextStyle(
            size=14
        ),
    )

    def on_select_plot(e):
        nonlocal image_path
        if selected_plot.value == 'نمودار تغییر نرخ خرابی':
            image_path = plot_failure_rate_change(results)
        elif selected_plot.value == 'نمودار نرخ کشف خرابی':
            image_path = plot_failure_detection_rate(results)
            
        image_control.src = image_path
        page.update()
    selected_plot.on_change = on_select_plot

    selected_model = Dropdown(
        label="انتخاب مدل تخمین قابلیت اطمینان",
        options=[
            dropdown.Option('مدل Goel Okumoto'),
            dropdown.Option('مدل Weibull'),
            dropdown.Option('مدل Log-Logistics'),
            dropdown.Option('مدل Duane'),
        ],
        text_align='right',
        text_style=TextStyle(
            size=14
        ),
        width=500
    )

    operational_time = TextField(label="زمان عملیات سیستم", value="10", keyboard_type=KeyboardType.NUMBER, width=290)
    operational_time_unit = Dropdown(
        label="واحد زمان",
        options=[
            dropdown.Option("ثانیه"),
            dropdown.Option("دقیقه"),
            dropdown.Option("ساعت")
        ],
        value="ثانیه",
        width=200
    )

    reliability_tile = ListTile(
        title=Text(""),
        visible=False,
        bgcolor='#dfdfdf',
        width=500
    )

    mtbf_tile = ListTile(
        title=Text("", rtl=True),
        visible=False,
        bgcolor='#dfdfdf',
        width=500,
    )

    def calculate_reliability(e):
        reliability_tile.visible = False
        page.update()

        reliability = 0
        error = None
        if operational_time_unit.value == 'ساعت':
            operational_time_value = int(operational_time.value) * 60 * 60
        elif operational_time_unit.value == 'دقیقه':
            operational_time_value = int(operational_time.value) * 60
        else:
            operational_time_value = int(operational_time.value)

        if selected_model.value == 'مدل Goel Okumoto':
            reliability = estimate_goel_okumoto(results, operational_time_value)
        elif selected_model.value == 'مدل Weibull':
            reliability, error = estimate_weibull(results, operational_time_value)
        elif selected_model.value == 'مدل Log-Logistics':
            reliability = estimate_log_logistics(results, operational_time_value)
        elif selected_model.value == 'مدل Duane':
            reliability = estimate_duane(results, operational_time_value)
        
        if error:
            reliability_tile.title.value = error
        else:
            reliability_tile.title.value = f"قابلیت اطمینان سیستم: {reliability:.4f}"
        reliability_tile.visible = True
        page.update()

    def calculate_mtbf(e):
        mtbf_tile.visible = False
        page.update()

        total_time = results[-1]['cumulative_time']
        total_failures = results[-1]['cumulative_failures']
        mtbf = float(total_time) / total_failures
        if operational_time_unit.value == 'ساعت':
            mtbf = mtbf / 3600
        elif operational_time_unit.value == 'دقیقه':
            mtbf = mtbf / 60
        
        mtbf_tile.title.value = f"شاخص میانگین زمان بین خرابی‌ها (MTBF): {mtbf:.4f} {operational_time_unit.value}"
        mtbf_tile.visible = True
        page.update()

    return Column([
        Container(
            content=Text("محاسبه قابلیت اطمینان با مدل‌های رشد", style=TextThemeStyle.HEADLINE_MEDIUM),
            alignment=alignment.center,
            padding=30
        ),
        Row([
            Container(
                content=Column([table], scroll=ScrollMode.AUTO),
                height=400,
                bgcolor=Colors.GREY_100,
                padding=10,
                border_radius=10,
                border=border.all(1, Colors.GREY_300),
            ),
            Column([
                selected_plot,
                image_control
            ], expand=True, height=400)
        ], expand=True),
        Row([
            operational_time,
            operational_time_unit
        ], width=500, alignment='center'),
        selected_model,
        Row([
            ElevatedButton(
                text="محاسبه قابلیت اطمینان",
                bgcolor=Colors.BLUE_500,
                color=Colors.WHITE,
                style=ButtonStyle(
                    shape= RoundedRectangleBorder(8),
                    padding=Padding(15, 15, 15, 15)
                ),
                on_click=calculate_reliability
            ),
            ElevatedButton(
                text="محاسبه شاخص MTBF",
                bgcolor=Colors.BLUE_400,
                color=Colors.WHITE,
                style=ButtonStyle(
                    shape= RoundedRectangleBorder(8),
                    padding=Padding(15, 15, 15, 15)
                ),
                on_click=calculate_mtbf
            )
        ], expand=True, alignment='center'),
        reliability_tile,
        mtbf_tile,
    ], horizontal_alignment='center')

def build_tab_test_and_estimation_model_run_tests(page: Page):
    all_testcases = load_all_testcases()

    number_of_failures = 0
    total_execution_time = 0
    running_threads = []
    thread_statuses = Column(width=600, spacing=10, scroll=ScrollMode.AUTO)

    def start_tester_test(testerId):
        nonlocal number_of_failures, running_threads

        number_of_self_failures = 0

        iteration = 1

        while(True):
            for test_case in all_testcases:
                thread_statuses.controls[testerId].subtitle = Text(f"در حال ساخت داده آزمون برای سناریوی {test_case}")
                page.update()

                number_of_sub_tests = math.ceil((int(number_of_tests.value) * all_testcases[test_case]['percent']) / 100)
                tester_iteration = f"{testerId + 1}-{iteration}"
                test_case_path = all_testcases[test_case]['testcase_dir']
                csv_path = generate_input_data(test_case, test_case_path, tester_iteration, number_of_sub_tests, False)
                if not csv_path:
                    continue

                df = pd.read_csv(csv_path)
                df['result'] = ''

                for idx, row in df.iterrows():
                    thread_statuses.controls[testerId].subtitle = Text(f"سناریوی {test_case}: در حال اجرای آزمون {idx+1}ام از {number_of_sub_tests} آزمون")
                    page.update()

                    env = os.environ.copy()
                    env.update(row.dropna().astype(str).to_dict())
                    try:
                        result = subprocess.run(
                            ["python", test_case_path],
                            env=env,
                            capture_output=True,
                            text=True, timeout=300
                        )
                        output = result.stdout.strip().splitlines()
                        outcome = next((line.strip() for line in output if line.strip() in ("pass", "fail")), "fail")

                        df.at[idx, "result"] = outcome
                        df.to_csv(csv_path, index=False)

                        if outcome == 'fail':
                            number_of_failures += 1
                            number_of_self_failures += 1
                    except Exception as e:
                        print(e)
                        number_of_failures += 1
                        number_of_self_failures += 1

                    if time.time() >= total_execution_time:
                        df = df.iloc[:idx+1]
                        df.to_csv(csv_path, index=False)
                        break

                if time.time() >= total_execution_time:
                        break
                
            if time.time() >= total_execution_time:
                        break
            
            iteration += 1

        running_threads[testerId] = False
        thread_statuses.controls[testerId].subtitle = Text(f"انجام آزمون با مشاهده {number_of_self_failures} خطا پایان یافت")
        thread_statuses.controls[testerId].trailing = Icon(name=Icons.DONE, color='Green')
        page.update()

    def run_testcase(e):
        nonlocal running_threads, number_of_failures, total_execution_time
        number_of_failures = 0
        running_threads = [False] * int(number_of_testers.value)
        thread_statuses.controls.clear()
        start_tests_button.disabled = True

        for i in range(int(number_of_testers.value)):
            thread_statuses.controls.append(
                ListTile(
                    title=Text(f"آزمونگر {i+1}"),
                    subtitle=Text("در حال آماده سازی..."),
                    trailing=ProgressRing(width=15, height=15),
                    bgcolor="#dfdfdf"
                )
            )
            t = threading.Thread(target=start_tester_test, args=(i,), daemon=True)
            t.start()
            running_threads[i] = True
        
        page.update()

        while (True in running_threads):
            pass

        start_tests_button.disabled = False
        
        reliability = 0
        if operational_time_unit.value == 'ساعت':
            operational_time_value = int(operational_time.value) * 60 * 60
        elif operational_time_unit.value == 'دقیقه':
            operational_time_value = int(operational_time.value) * 60
        else:
            operational_time_value = int(operational_time.value)

        if operational_time_unit.value == 'ساعت':
            total_execution_time = int(test_duration.value) * 60 * 60
        elif operational_time_unit.value == 'دقیقه':
            total_execution_time = int(test_duration.value) * 60
        else:
            total_execution_time = int(test_duration.value)

        all_testers_time = total_execution_time  * int(number_of_testers.value)

        all_testers_time_formatted = f"{all_testers_time // 60:02}:{all_testers_time % 60:02}"
        thread_statuses.controls.append(
            ListTile(
                title=Text("پایان فرایند آزمون"),
                subtitle=Text(f"تعداد کل خرابی‌ها: {number_of_failures}\nزمان سپری شده: {all_testers_time_formatted}"),
                bgcolor="#84c5fd"
            )
        )
        
        reliability = test_and_estimation_reliability(number_of_failures, all_testers_time, operational_time_value)
        reliability_tile.title.value = f"قابلیت اطمینان سیستم: {reliability:.4f}"
        reliability_tile.visible = True

        mtbf = float(all_testers_time) / number_of_failures if number_of_failures != 0 else 0
        if operational_time_unit.value == 'ساعت':
            mtbf = mtbf / 3600
        elif operational_time_unit.value == 'دقیقه':
            mtbf = mtbf / 60
        
        mtbf_tile.title.value = f"شاخص میانگین زمان بین خرابی‌ها (MTBF): {mtbf:.4f} {operational_time_unit.value}"
        mtbf_tile.visible = True
        page.update()

    test_duration = TextField(label="مقدار زمان آزمون", value="10", keyboard_type=KeyboardType.NUMBER)
    number_of_tests = TextField(label="تعداد آزمون", value="10", keyboard_type=KeyboardType.NUMBER)
    number_of_testers = TextField(label="تعداد آزمونگرها", value="1", keyboard_type=KeyboardType.NUMBER)

    operational_time = TextField(label="زمان عملیات سیستم", value="10", keyboard_type=KeyboardType.NUMBER)
    operational_time_unit = Dropdown(
        label="واحد زمان",
        options=[
            dropdown.Option("ثانیه"),
            dropdown.Option("دقیقه"),
            dropdown.Option("ساعت")
        ],
        value="ثانیه"
    )

    reliability_tile = ListTile(
        title=Text(""),
        visible=False,
        bgcolor='#dfdfdf',
        width=600
    )

    mtbf_tile = ListTile(
        title=Text("", rtl=True),
        visible=False,
        bgcolor='#dfdfdf',
        width=600,
    )

    start_tests_button = ElevatedButton(
        text="اجرای آزمون‌ها و محاسبه قابلیت اطمینان",
        bgcolor=Colors.BLUE_500,
        color=Colors.WHITE,
        style=ButtonStyle(
            shape= RoundedRectangleBorder(8),
            padding=Padding(15, 15, 15, 15)
        ),
        on_click=run_testcase
    )

    return Column([
        Container(
            content=Text("اجرای آزمون‌ها و محاسبه قابلیت اطمینان", style=TextThemeStyle.HEADLINE_MEDIUM),
            alignment=alignment.center,
            padding=30
        ),
        Column([
            test_duration,
            number_of_tests,
            number_of_testers,
            Row([
                operational_time,
                operational_time_unit
            ], expand=True),
            start_tests_button
        ], width=450, horizontal_alignment='center'),
        thread_statuses,
        reliability_tile,
        mtbf_tile
    ], expand=True, horizontal_alignment='center')

def build_tab_test_and_estimation_calculate_test_time(page: Page):
    consumer_risk_percent = Slider(min=0, max=100, divisions=100, label="{value}%", value=0)
    producer_risk_percent = Slider(min=0, max=100, divisions=100, label="{value}%", value=0)
    time_unit = Dropdown(
        label="واحد زمان",
        options=[
            dropdown.Option("ثانیه"),
            dropdown.Option("دقیقه"),
            dropdown.Option("ساعت")
        ],
        value="ثانیه"
    )
    ideal_mtbf = TextField(label="میانگین زمان بین خرابی مطلوب", value="10", keyboard_type=KeyboardType.NUMBER)
    minimum_mtbf = TextField(label="پایین ترین مقدار زمان بین خرابی مطلوب", value="10", keyboard_type=KeyboardType.NUMBER)

    test_plan = ListTile(
        title=Text(""),
        bgcolor='#dfdfdf',
        width=500,
        visible=False
    )

    def find_test_plan(e, max_c=100, max_T=100000, step=50):
        alpha = consumer_risk_percent.value / 100.0
        beta = producer_risk_percent.value / 100.0
        theta0 = int(ideal_mtbf.value)
        theta1 = int(minimum_mtbf.value)
        test_plan.visible = True
        test_plan.subtitle = ProgressBar(width=100, visible=True)
        test_plan.title.value = "در حال محاسبه، لطفا صبور باشید..."
        page.update()

        for c in range(1, max_c + 1):
            for T in np.arange(0, max_T, step):
                mu1 = T / theta1
                mu0 = T / theta0

                p_accept_H1 = poisson.cdf(c, mu1)
                p_accept_H0 = poisson.cdf(c, mu0)

                if p_accept_H0 >= 1 - beta and p_accept_H1 <= alpha:
                    test_plan.title.value = f"برنامه آزمون"
                    test_plan.subtitle = Text(f"بیشترین تعداد خرابی مجاز: {c}\nکمترین مقدار زمان آزمون: {round(T, 2)} {time_unit.value}")
                    page.update()
                    return
                
        test_plan.subtitle.visible = False
        test_plan.title.value = "متاسفانه برنامه آزمون مناسبی پیدا نشد."
        page.update()

    return Column([
        Container(
            content=Text("محاسبه زمان مورد نیاز برای آزمون", style=TextThemeStyle.HEADLINE_MEDIUM),
            alignment=alignment.center,
            padding=30
        ),
        Column([
            Row([
                Text("ریسک مصرف کننده: "),
                consumer_risk_percent,
            ], width=200),
            Row([
                Text("ریسک تولید کننده: "),
                producer_risk_percent,
            ], width=200),
            time_unit,
            ideal_mtbf,
            minimum_mtbf,
            ElevatedButton(
                text="محاسبه",
                bgcolor=Colors.BLUE_500,
                color=Colors.WHITE,
                style=ButtonStyle(
                    shape= RoundedRectangleBorder(8),
                    padding=Padding(15, 15, 15, 15)
                ),
                on_click=find_test_plan
            )
        ], width=450, horizontal_alignment='center'),
        test_plan
    ], expand=True, horizontal_alignment='center')

def build_tab_test_and_estimation_modify_results(page: Page):
    csv_files_list = ListView(spacing=10, padding=20, auto_scroll=True, width=400)
    csv_directory = 'test_and_estimate'
    files = os.listdir(csv_directory)
    for file_name in files:
        if file_name.lower().endswith(".csv"):
            csv_files_list.controls.append(
                ListTile(
                    title=Text(file_name),
                    trailing=Icon(name="EDIT", color="blue"),
                    on_click=lambda e, name=file_name: select_results_file(name)
                )
            )

    data_table = DataTable(columns=[DataColumn(Text("داده‌ای برای نمایش وجود ندارد"))])
    df = pd.DataFrame()
    selected_csv_path = ''
    
    def select_results_file(filename):
        nonlocal data_table, df, selected_csv_path

        save_message.value = ''
        selected_csv_path = os.path.join(csv_directory, filename)
        df = pd.read_csv(os.path.join(csv_directory, filename))
        columns = [DataColumn(Text(col)) for col in df.columns]

        rows = []
        for index, row in df.iterrows():
            cells = [
                DataCell(Text(str(row[col]))) if col != 'result'
                else DataCell(
                    Dropdown(
                        options=[
                            dropdown.Option("پاس شده"),
                            dropdown.Option("شکست خورده"),
                        ],
                        value="پاس شده" if str(row[col]) == "pass" else "شکست خورده",
                        on_change=lambda e, idx=index: modify_test_result(e, idx)
                    )
                )
                for col in df.columns
            ]

            rows.append(DataRow(cells=cells))

        data_table.columns = columns
        data_table.rows = rows

        page.update()

        def modify_test_result(e, row_index):
            save_message.value = ''
            page.update()

            modified_result = 'pass' if e.control.value == 'پاس شده' else 'fail' 
            df.at[row_index, 'result'] = modified_result
            
            
    def save_modified_results(e):
        if df.empty:
            save_message.value = 'هیچ فایلی انتخاب نشده است'
            page.update()
            return
        df.to_csv(selected_csv_path, index=False)
        
        save_message.value = 'تغییرات با موفقیت ذخیره شد'
        page.update()

    def calculate_reliability(e):
        total_failures = 0
        for filename in os.listdir(csv_directory):
            if filename.endswith('.csv'):
                file_path = os.path.join(csv_directory, filename)
                try:
                    data = pd.read_csv(file_path)
                    fail_rows = data[data['result'].astype(str).str.contains('fail', case=False, na=False)]
                    total_failures += len(fail_rows)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        reliability = 0
        if operational_time_unit.value == 'ساعت':
            operational_time_value = int(operational_time.value) * 60 * 60
        elif operational_time_unit.value == 'دقیقه':
            operational_time_value = int(operational_time.value) * 60
        else:
            operational_time_value = int(operational_time.value)

        if operational_time_unit.value == 'ساعت':
            total_execution_time = int(test_duration.value) * 60 * 60
        elif operational_time_unit.value == 'دقیقه':
            total_execution_time = int(test_duration.value) * 60
        else:
            total_execution_time = int(test_duration.value)
        
        reliability = test_and_estimation_reliability(total_failures, total_execution_time, operational_time_value)
        reliability_tile.title.value = f"قابلیت اطمینان سیستم: {reliability:.4f}"
        reliability_tile.visible = True

        mtbf = float(total_execution_time) / total_failures
        if operational_time_unit.value == 'ساعت':
            mtbf = mtbf / 3600
        elif operational_time_unit.value == 'دقیقه':
            mtbf = mtbf / 60
        
        mtbf_tile.title.value = f"شاخص میانگین زمان بین خرابی‌ها (MTBF): {mtbf:.4f} {operational_time_unit.value}"
        mtbf_tile.visible = True
        page.update()

    test_duration = TextField(label="مقدار زمان آزمون", value="10", keyboard_type=KeyboardType.NUMBER, width=200)
    operational_time = TextField(label="زمان عملیات سیستم", value="10", keyboard_type=KeyboardType.NUMBER)
    operational_time_unit = Dropdown(
        label="واحد زمان",
        options=[
            dropdown.Option("ثانیه"),
            dropdown.Option("دقیقه"),
            dropdown.Option("ساعت")
        ],
        value="ثانیه"
    )

    save_message = Text("")

    reliability_tile = ListTile(
        title=Text(""),
        visible=False,
        bgcolor='#dfdfdf',
        width=500
    )

    mtbf_tile = ListTile(
        title=Text("", rtl=True),
        visible=False,
        bgcolor='#dfdfdf',
        width=500,
    )

    return Column([
        Container(
            content=Text("اصلاح دستی نتایج", style=TextThemeStyle.HEADLINE_MEDIUM),
            alignment=alignment.center,
            padding=30
        ),
        Row([
            Column([
                Text('لیست فایل‌های نتایج'),
                csv_files_list,
                ElevatedButton(
                    text="ذخیره تغییرات فایل انتخاب شده",
                    bgcolor=Colors.BLUE_500,
                    color=Colors.WHITE,
                    style=ButtonStyle(
                        shape= RoundedRectangleBorder(8),
                        padding=Padding(15, 15, 15, 15)
                    ),
                    on_click=save_modified_results
                ),
                save_message
            ], horizontal_alignment='center', width=400),
            Container(
                content=Column([data_table], scroll=ScrollMode.AUTO),
                height=400,
                bgcolor=Colors.GREY_100,
                padding=10,
                border_radius=10,
                border=border.all(1, Colors.GREY_300),
            ),
        ], expand=True, alignment='center'),
        test_duration,
        Row([
            operational_time,
            operational_time_unit
        ], expand=True, alignment='center', width=500),
        ElevatedButton(
            text="محاسبه قابلیت اطمینان",
            bgcolor=Colors.BLUE_500,
            color=Colors.WHITE,
            style=ButtonStyle(
                shape= RoundedRectangleBorder(8),
                padding=Padding(15, 15, 15, 15)
            ),
            on_click=calculate_reliability
        ),
        reliability_tile,
        mtbf_tile
    ], expand=True, horizontal_alignment='center')

def main(page: Page):
    page.title = "ماژول محاسبه‌گر قابلیت اطمینان نرم‌افزار"
    page.scroll = ScrollMode.AUTO
    page.rtl = True

    growth_method_tabs = Tabs(tabs=[
        Tab(text="اجرای آزمون‌ها", content=build_tab_growth_model_run_tests(page)),
        Tab(text="محاسبه قابلیت اطمینان", content=build_tab_growth_reliability(page)),
    ])

    test_and_estimation_method_tabs = Tabs(tabs=[
        Tab(text="محاسبه زمان مورد نیاز برای آزمون", content=build_tab_test_and_estimation_calculate_test_time(page)),
        Tab(text="اجرای آزمون‌ها و محاسبه قابلیت اطمینان", content=build_tab_test_and_estimation_model_run_tests(page)),
        Tab(text="اصلاح دستی نتایج", content=build_tab_test_and_estimation_modify_results(page)),
    ])

    page.add(Tabs(tabs=[
        Tab(text="مدیریت آزمون‌ها", content=build_tab_manage_tests(page)),
        Tab(text="محاسبه قابلیت اطمینان با مدل‌های رشد", content=growth_method_tabs),
        Tab(text="محاسبه قابلیت اطمینان با مدل تست و تخمین", content=test_and_estimation_method_tabs)
    ]))

app(target=main)
