import os
import json
from flet import *
import subprocess
import csv
import statistics
import shutil

PROJECT_CONFIG = 'project_config.json'
GROWTH_RESULTS_FILE = 'results.json'

def load_prject_config():
    if os.path.exists(PROJECT_CONFIG):
        with open(PROJECT_CONFIG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {'selected_project': 'default', 'jmeter_path': '', 'sikulix_path': '', 'projects': []}

def load_results(project_name):
    if os.path.exists(f"web/{project_name}/{GROWTH_RESULTS_FILE}"):
        with open(f"web/{project_name}/{GROWTH_RESULTS_FILE}", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_results(data, project_name):
    with open(f"web/{project_name}/{GROWTH_RESULTS_FILE}", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def build_tab_web_load_test(page: Page):
    project_config = load_prject_config()
    project_name = project_config['selected_project']
    os.makedirs(f"web/{project_name}/loadtest", exist_ok=True)

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
                    allowed_extensions=['jmx'],
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

    number_of_threads = TextField(label="تعداد تردها (حجم بار)", value="100", keyboard_type=KeyboardType.NUMBER)
    ramp_up_period = TextField(label="مقدار زمان لازم برای ایجاد تردها (ثانیه)", value="30", keyboard_type=KeyboardType.NUMBER)
    loop_count = TextField(label="تعداد درخواست های هر ترد به هر api", value="30", keyboard_type=KeyboardType.NUMBER)

    error_percent_field = TextField(
        label="حداکثر درصد خطای قابل قبول",
        value="0",
        suffix_text="%",
        keyboard_type=KeyboardType.NUMBER,
        max_length=6,
    )

    def on_error_percent_field_change(e):
        raw = e.control.value
        # Keep only digits and one dot
        filtered = ""
        dot_seen = False
        for ch in raw:
            if ch.isdigit():
                filtered += ch
            elif ch == "." and not dot_seen:
                filtered += ch
                dot_seen = True
        if filtered != raw:
            error_percent_field.value = filtered
            page.update()

    error_percent_field.on_change = on_error_percent_field_change

    max_p99_field = TextField(
        label="حداکثر صدک 99 قابل قبول",
        value="500",
        suffix_text="میلی ثانیه",
        keyboard_type=KeyboardType.NUMBER,
        max_length=4,
    )

    def on_max_p99_field_change(e):
        raw = e.control.value
        # Keep only digits
        filtered = ""
        for ch in raw:
            if ch.isdigit():
                filtered += ch
        if filtered != raw:
            max_p99_field.value = filtered
            page.update()
    
    max_p99_field.on_change = on_max_p99_field_change

    load_status_tile = ListTile(
        title=Text("در حال اجرای تست بار"),
        subtitle=Text("لطفا منتظر بمانید..."),
        trailing=ProgressRing(width=15, height=15),
        bgcolor="#dfdfdf",
        width=600,
        visible=False
    )

    load_metrics_tile = ListTile(
        title=Text("", rtl=True),
        visible=False,
        bgcolor='#dfdfdf',
        width=600,
    )

    test_result_tile = ListTile(
        title=Text(""),
        subtitle=Text(""),
        bgcolor="#84c5fd",
        width=600,
        visible=False,
    )

    def run_load_test():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, f"web/{project_name}/loadtest")
        jmx_name = os.path.splitext(os.path.basename(testcase_dir_input.value))[0]
        jtl_path = os.path.join(results_dir, f"{jmx_name}.jtl")
        report_dir = os.path.join(results_dir, f"{jmx_name}_report")

        if os.path.exists(jtl_path):
            os.remove(jtl_path)

        if os.path.exists(report_dir):
            shutil.rmtree(report_dir)

        jmeter_path = project_config["jmeter_path"]

        cmd = [
            jmeter_path,
            "-n",
            "-t", testcase_dir_input.value,
            "-Jthreads=" + number_of_threads.value,
            "-Jrampup=" + ramp_up_period.value,
            "-Jloop=" + loop_count.value,
            "-l", jtl_path
        ]

        subprocess.run(cmd)

        report_cmd = [
            jmeter_path,
            "-g", jtl_path,
            "-o", report_dir
        ]

        subprocess.run(report_cmd)

        return jtl_path

    def check_test_failure(error, p99) -> bool:
        return error > float(error_percent_field.value) or p99 > float(max_p99_field.value)

    def parse_jtl_csv(jtl_path):
        timestamps = []
        response_times = []
        total_requests = 0
        failed_requests = 0

        with open(jtl_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                total_requests += 1

                success = row.get("success", "").lower() == "true"
                if not success:
                    failed_requests += 1

                ts = int(row.get("timeStamp", "0"))
                elapsed = int(row.get("elapsed", "0"))

                timestamps.append(ts)
                timestamps.append(ts + elapsed)

                response_times.append(elapsed)

        if timestamps:
            test_duration_ms = max(timestamps) - min(timestamps)
            test_duration_sec = test_duration_ms / 1000
        else:
            test_duration_sec = 0

        error_rate = failed_requests / total_requests if total_requests else 0
        error_rate = error_rate * 100.0

        if response_times:
            avg_response = statistics.mean(response_times)
            max_response = max(response_times)
            min_response = min(response_times)
            median_response = statistics.median(response_times)

            sorted_times = sorted(response_times)

            def percentile(p):
                if not sorted_times:
                    return 0
                k = (len(sorted_times) - 1) * (p / 100)
                f = int(k)
                c = min(f + 1, len(sorted_times) - 1)
                return sorted_times[f] + (sorted_times[c] - sorted_times[f]) * (k - f)

            p90 = percentile(90)
            p95 = percentile(95)
            p99 = percentile(99)

        else:
            avg_response = max_response = min_response = median_response = 0
            p90 = p95 = p99 = 0

        throughput = total_requests / test_duration_sec if test_duration_sec > 0 else 0

        is_test_failed = check_test_failure(error=error_rate, p99=p99)

        return {
            "تعداد کل درخواست‌ها": total_requests,
            "تعداد خطاها": failed_requests,
            "درصد خطا": f"{round(error_rate, 2)}%",
            "زمان کل تست به ثانیه": test_duration_sec,
            "میانگین زمان پاسخ": f"{round(avg_response, 2)} میلی ثانیه",
            "کمترین زمان پاسخ": f"{round(min_response, 2)} میلی ثانیه",
            "بیشترین زمان پاسخ": f"{round(max_response, 2)} میلی ثانیه",
            "میانه زمان پاسخ": f"{round(median_response, 2)} میلی ثانیه",
            "معیار صدک 90": f"{round(p90, 2)} میلی ثانیه",
            "معیار صدک 95": f"{round(p95, 2)} میلی ثانیه",
            "معیار صدک 99": f"{round(p99, 2)} میلی ثانیه",
            "تعداد درخواست پاسخ داده شده در ثانیه": f"{round(throughput, 2)} درخواست",
        }, is_test_failed
    
    def perform_load_test(e):
        load_status_tile.visible = True
        load_metrics_tile.visible = False
        test_result_tile.visible = False
        page.update()

        jtl_path = run_load_test()
        load_test_results, is_test_failed = parse_jtl_csv(jtl_path)

        load_status_tile.visible = False
        load_results_str = "\n".join([f"{k}: {v}" for k, v in load_test_results.items()])
        load_metrics_tile.title.value = load_results_str
        load_metrics_tile.visible = True

        number_of_failures = 0
        total_execution_time = int(load_test_results["زمان کل تست به ثانیه"])

        if is_test_failed:
            test_result_tile.title = Text("شکست آزمون")
            test_result_tile.bgcolor = "#ff0d05"
            test_result_tile.trailing = Icon(name=Icons.CANCEL)
            number_of_failures = 1
        else:
            test_result_tile.title = Text("موفقیت آزمون")
            test_result_tile.bgcolor = "#05ff2f"
            test_result_tile.trailing = Icon(name=Icons.DONE)

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
        page.pubsub.send_all('web_growth_results')

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
            ramp_up_period,
            loop_count,
            error_percent_field,
            max_p99_field,
            start_tests_button
        ], width=450, horizontal_alignment='center'),
        load_status_tile,
        load_metrics_tile,
        test_result_tile,
    ], spacing=30, expand=True, horizontal_alignment='center')