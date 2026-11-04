import os
import json
from flet import *
import subprocess
import time
import re
from Utils import add_desktop_stress_loop

PROJECT_CONFIG = 'project_config.json'
GROWTH_RESULTS_FILE = 'results.json'

def load_prject_config():
    if os.path.exists(PROJECT_CONFIG):
        with open(PROJECT_CONFIG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {'selected_project': 'default', 'jmeter_path': '', 'sikulix_path': '', 'projects': []}

def load_results(project_name):
    if os.path.exists(f"desktop/{project_name}/{GROWTH_RESULTS_FILE}"):
        with open(f"desktop/{project_name}/{GROWTH_RESULTS_FILE}", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_results(data, project_name):
    with open(f"desktop/{project_name}/{GROWTH_RESULTS_FILE}", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def build_tab_desktop_load_test(page: Page):
    project_config = load_prject_config()
    project_name = project_config['selected_project']
    os.makedirs(f"desktop/{project_name}/loadtest", exist_ok=True)

    testcase_dir_picker = FilePicker()
    page.overlay.append(testcase_dir_picker)
    testcase_dir_input = TextField(
        label="📂 مسیر فایل سناریو آزمون",
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
                on_click=lambda e: testcase_dir_picker.pick_files(
                    file_type=FilePickerFileType.CUSTOM,
                    allowed_extensions=['py'],
                    allow_multiple=False
                )
            )
        ],
        spacing=10
    )

    def on_file_selected(e):
        if e.files:
            testcase_dir_input.value = e.files[0].path
            testcase_dir_input.update()

    testcase_dir_picker.on_result = on_file_selected

    number_of_threads = TextField(label="تعداد دفعات اجرا (حجم بار)", value="100", keyboard_type=KeyboardType.NUMBER)
    test_duration = TextField(label="حداکثر زمان اجرای آزمون (ثانیه)", value="30", keyboard_type=KeyboardType.NUMBER)


    load_status_tile = ListTile(
        title=Text("در حال اجرای تست بار"),
        subtitle=Text("لطفا منتظر بمانید..."),
        trailing=ProgressRing(width=15, height=15),
        bgcolor="#dfdfdf",
        width=600,
        visible=False
    )

    test_result_tile = ListTile(
        title=Text(""),
        subtitle=Text(""),
        bgcolor="#84c5fd",
        width=600,
        visible=False,
    )

    def run_load_test():
        iterations = int(number_of_threads.value)
        add_desktop_stress_loop(
            file_path=testcase_dir_input.value,
            iterations=iterations,
        )

        start_time = time.perf_counter()

        try:
            process_env = os.environ.copy()
            process_env["SIKULIX_ITERATIONS"] = str(iterations)

            result = subprocess.run(
                [
                    "java",
                    "-jar",
                    project_config["sikulix_path"],
                    "-c",
                    "-r",
                    testcase_dir_input.value,
                ],
                env=process_env,
                capture_output=True,
                text=True,
                timeout=int(test_duration.value),
            )

            elapsed = time.perf_counter() - start_time

            output = (result.stdout or "") + "\n" + (result.stderr or "")

            completed = len(re.findall(
                r"__ITERATION_COMPLETED__:\d+",
                output,
            ))

            passed = (
                True
                if result.returncode == 0 and completed == iterations
                else False
            )

            return passed, int(elapsed)

        except subprocess.TimeoutExpired as exc:

            elapsed = time.perf_counter() - start_time

            stdout = exc.stdout or ""
            stderr = exc.stderr or ""

            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")

            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")

            output = stdout + "\n" + stderr

            completed = len(re.findall(
                r"__ITERATION_COMPLETED__:\d+",
                output,
            ))

            passed = (
                True
                if completed == iterations
                else False
            )

            return passed, int(elapsed)

            
        
    def perform_load_test(e):
        load_status_tile.visible = True
        test_result_tile.visible = False
        page.update()

        passed, elapsed = run_load_test()

        load_status_tile.visible = False

        number_of_failures = 0
        total_execution_time = elapsed

        if passed:
            test_result_tile.title = Text("موفقیت آزمون")
            test_result_tile.subtitle = Text(f"زمان سپری شده: {elapsed} ثانیه")
            test_result_tile.bgcolor = "#05ff2f"
            test_result_tile.trailing = Icon(name=Icons.DONE)
        else:
            test_failure_cause = "عبور از زمان تعیین شده" if elapsed > int(test_duration.value) else "خطا در فرایند آزمون"
            test_result_tile.title = Text("شکست آزمون")
            test_result_tile.subtitle = Text(f"زمان سپری شده: {elapsed} ثانیه\nدلیل شکست آزمون: {test_failure_cause}")
            test_result_tile.bgcolor = "#ff0d05"
            test_result_tile.trailing = Icon(name=Icons.CANCEL)
            number_of_failures = 1
            

        test_result_tile.visible = True

        page.update()

        if total_execution_time == 0:
            return
        
        results = load_results(project_name)
        last_result = results[-1] if results else {"cumulative_failures": 0, "cumulative_time": 0}
        cumulative_failures = last_result["cumulative_failures"] + number_of_failures
        cumulative_time = last_result["cumulative_time"] + total_execution_time
        results.append({
            "failures": number_of_failures,
            "time": total_execution_time,
            "cumulative_failures": cumulative_failures,
            "cumulative_time": cumulative_time,
            "failure_rate": cumulative_failures / cumulative_time
        })
        save_results(results, project_name)
        page.pubsub.send_all('desktop_growth_results')

    start_tests_button = ElevatedButton(
        text="شروع آزمون",
        bgcolor=Colors.BLUE_500,
        color=Colors.WHITE,
        style=ButtonStyle(
            shape= RoundedRectangleBorder(8),
            padding=Padding(15, 15, 15, 15)
        ),
        on_click=perform_load_test
    )

    return Column([
        Container(
            content=Text("آزمون بار و استرس", size=20, weight="bold", text_align="center"),
            alignment=alignment.center,
            padding=30
        ),
        Column([
            testcase_dir_row,
            number_of_threads,
            test_duration,
            start_tests_button
        ], width=450, horizontal_alignment='center'),
        load_status_tile,
        test_result_tile,
    ], spacing=30, expand=True, horizontal_alignment='center')