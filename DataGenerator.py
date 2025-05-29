import json
import os
from flet import *
import time

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


def main(page: Page):
    page.title = "سیستم مدیریت و اجرای آزمون"
    page.scroll = ScrollMode.AUTO
    page.rtl = True

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

    def start_tester_test(testerId):
        pass
        # number_of_tests_before_first_failure = 0
        # start_time = time.time()

        # for test in tests:
        #     n_rows = settings['total_number_of_tests'] * (tests[test]['percent'] / 100.0)
        #     df = generate_data(tests[test]["fields"], math.ceil(n_rows))
        #     df['result'] = ''
        #     csv_name = tests[test]["testcase_name"].replace('.py', '') + '-tester' + str(testerId) + '.csv'
        #     csv_path = os.path.join(tests[test]["testcase_dir"], csv_name)
        #     df.to_csv(csv_path, index=False)

        #     for idx, row in df.iterrows():
        #         env = os.environ.copy()
        #         env.update(row.dropna().astype(str).to_dict())
        #         try:
        #             result = subprocess.run(
        #                 ["python", os.path.join(tests[test]["testcase_dir"], tests[test]["testcase_name"])],
        #                 env=env,
        #                 capture_output=True,
        #                 text=True, timeout=300
        #             )
        #             output = result.stdout.strip().splitlines()
        #             outcome = next((line.strip() for line in output if line.strip() in ("pass", "fail")), "fail")

        #             df.at[idx, "result"] = outcome
        #             df.to_csv(csv_path, index=False)

        #             number_of_tests_before_first_failure += 1

        #             # tester_name.markdown(f"*Tester{testerId}*")
        #             # elapsed = int(time.time() - start_time)
        #             # elapsed_formatted = f"{elapsed // 60:02}:{elapsed % 60:02}"
        #             # time_placeholder.markdown(f"⏱ **Elapsed Time**: `{elapsed_formatted}`")
        #             # status_placeholder.markdown(f"🔄 Running **{test}** test `{idx + 1}/{math.ceil(n_rows)}`")
        #             # separator.markdown("---")

        #             elapsed = int(time.time() - start_time)
        #             elapsed_formatted = f"{elapsed // 60:02}:{elapsed % 60:02}"
        #             st.session_state.results[testerId] = {
        #                 'elapsed_formatted': f"⏱ **Elapsed Time**: `{elapsed_formatted}`",
        #                 'status': f"🔄 Running **{test}** test `{idx + 1}/{math.ceil(n_rows)}`",
        #                 'is_failed': outcome == 'fail'
        #             }

        #             if outcome == 'fail':
        #                 return number_of_tests_before_first_failure, True
        #         except Exception as e:
        #             return "", str(e)
                
        # # tester_name.markdown(f"*Tester{testerId}*")
        # # elapsed = int(time.time() - start_time)
        # # elapsed_formatted = f"{elapsed // 60:02}:{elapsed % 60:02}"
        # # time_placeholder.markdown(f"⏱ **Elapsed Time**: `{elapsed_formatted}`")
        # # status_placeholder.markdown(f"🔄 Running **{test}** test `{math.ceil(n_rows)}/{math.ceil(n_rows)}`")
        # # separator.markdown("---")
                
        # elapsed = int(time.time() - start_time)
        # elapsed_formatted = f"{elapsed // 60:02}:{elapsed % 60:02}"

        # return number_of_tests_before_first_failure, False

    def run_testcase():
        number_of_failures = 0
        total_execution_time = 0
        for i in range(number_of_testers):
            execution_time, isFailed = start_tester_test(i+1)
            total_number_of_tests_executed += number_of_tests
            if isFailed:
                number_of_failures += 1
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

    refresh_test_list()

    tab1 = Column([
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

    tab2 = Column([
        Text("اجرای آزمون‌ها", style=TextThemeStyle.HEADLINE_MEDIUM),
        number_of_tests,
        number_of_testers,
        ElevatedButton("اجرای آزمون‌ها", on_click={}),
    ])

    page.add(Tabs(tabs=[
        Tab(text="مدیریت آزمون‌ها و تولید داده", content=tab1),
        Tab(text="اجرای آزمون‌ها", content=tab2)
    ]))

app(target=main)
