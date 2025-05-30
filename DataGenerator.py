import json
import os
from flet import *
import time
import pandas as pd
import subprocess
import math
import threading
import matplotlib.pyplot as plt
import uuid

SETTINGS_FILE = 'testcases.json'
RESULTS_FILE = 'results.json'


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

def generate_scatter_image(data):
    x = [item['failure_rate'] for item in data]
    y = [item['cumulative_failures'] for item in data]

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel("Failure Rate")
    ax.set_ylabel("Cumulative Failures")
    ax.set_title("Failure Rate vs Cumulative Failures")

    filename = f"plot_{uuid.uuid4().hex}.png"
    plt.savefig(filename)
    plt.close(fig)

    return filename

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
                "انتخاب فایل",
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

    percent_input = Slider(min=0, max=100, divisions=100, label="{value}%", value=0)
    new_test_name_input = TextField(label="نام سناریو جدید")
    test_list_column = Column(scroll=ScrollMode.AUTO)

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
        refresh_test_list()

    def save_testcase(e):
        new_name = new_test_name_input.value
        if not new_name:
            page.snack_bar = SnackBar(Text("لطفاً نام سناریو آزمون را وارد کنید."))
            page.snack_bar.open = True
            page.update()
            return
        all_testcases[new_name] = {
            "testcase_dir": testcase_dir_input.value,
            "percent": int(percent_input.value),
        }
        save_all_testcases(all_testcases)
        page.snack_bar = SnackBar(Text(f"سناریو '{new_name}' ذخیره شد."))
        page.snack_bar.open = True
        refresh_test_list()

    def on_test_select(e):
        name = selected_test.value
        if name and name in all_testcases:
            data = all_testcases[name]
            new_test_name_input.value = name
            testcase_dir_input.value = data["testcase_dir"]
            percent_input.value = data["percent"]
        else:
            new_test_name_input.value = ""
            testcase_dir_input.value = ""
            percent_input.value = 0
        page.update()

    selected_test.on_change = on_test_select

    refresh_test_list()

    return Column([
        Container(
            content=Text("🗂️ مدیریت سناریوهای آزمون", size=20, weight="bold", text_align="center"),
            alignment=alignment.center,
            padding=30
        ),
        Row([
            Column([
                selected_test,
                Column([
                    Text("لیست سناریوها برای حذف 📋", style=TextThemeStyle.TITLE_MEDIUM),
                    test_list_column,
                ])
            ], width=300, horizontal_alignment="center", alignment="start", spacing=50),
            Column([
                new_test_name_input,
                testcase_dir_row,
                percent_input,
                ElevatedButton("ذخیره سناریو", on_click=save_testcase),
            ], horizontal_alignment="start")
        ], alignment='start', vertical_alignment='start', expand=True, spacing=50),
    ], spacing=50)

def build_tab_run_tests(page: Page):
    all_testcases = load_all_testcases()

    number_of_failures = 0
    total_execution_time = 0
    running_threads = []
    thread_statuses = Column()

    def start_tester_test(testerId):
        nonlocal number_of_failures, total_execution_time, running_threads
        start_time = time.time()

        for test_case in all_testcases:
            number_of_sub_tests = math.ceil((int(number_of_tests.value) * all_testcases[test_case]['percent']) / 100)
            csv_path = test_case + '-tester' + str(testerId+1) + '.csv'
            test_case_path = all_testcases[test_case]["testcase_dir"]
            df = pd.read_csv(csv_path)
            df['result'] = ''

            for idx, row in df.iterrows():
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

                    tester_status = f" -> Test ({test_case}) -- Excuted {idx + 1} tests from {number_of_sub_tests} tests"
                    thread_statuses.controls[testerId].value = f"Tester {testerId}: {tester_status}"
                    page.update()

                    if outcome == 'fail':
                        elapsed = int(time.time() - start_time)
                        elapsed_formatted = f"{elapsed // 60:02}:{elapsed % 60:02}"

                        total_execution_time += elapsed
                        number_of_failures += 1
                        running_threads[testerId] = False

                        thread_statuses.controls[testerId].value = f"Tester {testerId} failed at {idx+1}th test from {test_case} -- Elapsed Time: {elapsed_formatted}"
                        page.update()
                        return
                except Exception as e:
                    
                    print(e)

                    elapsed = int(time.time() - start_time)
                    elapsed_formatted = f"{elapsed // 60:02}:{elapsed % 60:02}"

                    total_execution_time += elapsed
                    number_of_failures += 1
                    running_threads[testerId] = False

                    thread_statuses.controls[testerId].value = f"Tester {testerId} failed at {idx+1}th test from {test_case} -- Elapsed Time: {elapsed_formatted}"
                    page.update()
                    return
            
        elapsed = int(time.time() - start_time)
        elapsed_formatted = f"{elapsed // 60:02}:{elapsed % 60:02}"

        total_execution_time += elapsed
        running_threads[testerId] = False

        thread_statuses.controls[testerId].value = f"Tester {testerId} excuted all tests without failure -- Elapsed Time: {elapsed_formatted}"
        page.update()

    def run_testcase(e):
        nonlocal running_threads, number_of_failures, total_execution_time
        total_execution_time = 0
        number_of_failures = 0
        running_threads = [False] * int(number_of_testers.value)
        thread_statuses.controls.clear()
        for i in range(int(number_of_testers.value)):
            thread_statuses.controls.append(Text(f"Thread {i}: Not started", size=18))
            t = threading.Thread(target=start_tester_test, args=(i,), daemon=True)
            t.start()
            running_threads[i] = True
        
        page.update()

        while (True in running_threads):
            pass

        thread_statuses.controls.append(Text("All Tests Finished"))
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

    return Column([
        Text("اجرای آزمون‌ها", style=TextThemeStyle.HEADLINE_MEDIUM),
        number_of_tests,
        number_of_testers,
        ElevatedButton("اجرای آزمون‌ها", on_click=run_testcase),
        thread_statuses
    ])

def build_tab_show_results(page: Page):
    results = load_results()

    rows = [
        DataRow(
            cells=[
                DataCell(Text(str(idx + 1))),
                DataCell(Text(str(item["failures"]))),
                DataCell(Text(str(item["time"]))),
                DataCell(Text(str(item["cumulative_failures"]))),
                DataCell(Text(str(item["cumulative_time"]))),
                DataCell(Text(f"{item['failure_rate']:.4f}"))
            ]
        )
        for idx, item in enumerate(results)
    ]

    operational_time = TextField(label="زمان عملیات سیستم", value="10", keyboard_type=KeyboardType.NUMBER)

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

    image_path = generate_scatter_image(results)
    image_control = Image(src=image_path, width=400, height=300)

    return Column([
        Text("محاسبه قابلیت اطمینان", style=TextThemeStyle.HEADLINE_MEDIUM),
        Row([
            Container(
                content=Column([table], scroll=ScrollMode.AUTO),
                height=300,
                bgcolor=Colors.GREY_100,
                padding=10,
                border_radius=10,
                border=border.all(1, Colors.GREY_300),
            ),
            image_control,
        ], expand=True),
        operational_time,
        ElevatedButton("محاسبه قابلیت اطمینان", on_click={}),
    ])

def main(page: Page):
    page.title = "سیستم مدیریت و اجرای آزمون"
    page.scroll = ScrollMode.AUTO
    page.rtl = True

    page.add(Tabs(tabs=[
        Tab(text="مدیریت آزمون‌ها و تولید داده", content=build_tab_manage_tests(page)),
        Tab(text="اجرای آزمون‌ها", content=build_tab_run_tests(page)),
        Tab(text="محاسبه قابلیت اطمینان", content=build_tab_show_results(page))
    ]))

app(target=main)
