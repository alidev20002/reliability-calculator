import os
import json
from flet import *
import subprocess
import csv
import statistics

PROJECT_CONFIG = 'project_config.json'

def load_prject_config():
    if os.path.exists(PROJECT_CONFIG):
        with open(PROJECT_CONFIG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {'selected_project': 'default', 'jmeter_path': '', 'sikulix_path': '', 'projects': []}

def save_project_config(data):
    with open(PROJECT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def build_tab_web_load_test(page: Page):
    project_config = load_prject_config()
    project_name = project_config['selected_project']
    os.makedirs(f"web/{project_name}/loadtest", exist_ok=True)

    jmeter_dir_picker = FilePicker()
    page.overlay.append(jmeter_dir_picker)
    jmeter_dir_input = TextField(
        value=project_config['jmeter_path'],
        label="📂 مسیر فایل اجرایی ابزار jmeter (jmeter.bat یا jmeter.sh)",
        read_only=True
    )

    jmeter_dir_row = Row(
        controls=[
            jmeter_dir_input,
            ElevatedButton(
                text="انتخاب فایل",
                icon=Icons.UPLOAD_FILE,
                bgcolor=Colors.BLUE_500,
                color=Colors.WHITE,
                style=ButtonStyle(
                    shape= RoundedRectangleBorder(8),
                    padding=Padding(15, 15, 15, 15)
                ),
                on_click=lambda e: jmeter_dir_picker.pick_files(
                    file_type=FilePickerFileType.CUSTOM,
                    allowed_extensions=['bat', 'sh'],
                    allow_multiple=False
                )
            )
        ],
        spacing=10
    )

    def on_jmeter_file_selected(e):
        if e.files:
            jmeter_dir_input.value = e.files[0].path
            jmeter_dir_input.update()
            project_config['jmeter_path'] = e.files[0].path
            save_project_config(project_config)

    jmeter_dir_picker.on_result = on_jmeter_file_selected

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

    def run_load_test():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(base_dir, f"web/{project_name}/loadtest")
        jmx_name = os.path.splitext(os.path.basename(testcase_dir_input.value))[0]
        jtl_path = os.path.join(results_dir, f"{jmx_name}.jtl")
        if os.path.exists(jtl_path):
            os.remove(jtl_path)
        jmeter_path = jmeter_dir_input.value
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

        return jtl_path

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
        }
    
    def perform_load_test(e):
        load_status_tile.visible = True
        load_metrics_tile.visible = False
        page.update()

        jtl_path = run_load_test()
        load_test_results = parse_jtl_csv(jtl_path)

        load_status_tile.visible = False
        load_results_str = "\n".join([f"{k}: {v}" for k, v in load_test_results.items()])
        load_metrics_tile.title.value = load_results_str
        load_metrics_tile.visible = True

        page.update()

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
            jmeter_dir_row,
            testcase_dir_row,
            number_of_threads,
            ramp_up_period,
            loop_count,
            start_tests_button
        ], width=450, horizontal_alignment='center'),
        load_status_tile,
        load_metrics_tile
    ], spacing=30, expand=True, horizontal_alignment='center')