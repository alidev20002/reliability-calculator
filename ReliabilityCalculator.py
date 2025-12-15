import os
from flet import *
import WebSystemTestTabs as webSys
import DesktopSystemTestTabs as deskSys
import LoadTestTabs as load
import ProjectManagementTab as pm

os.makedirs('SystemTest', exist_ok=True)
os.makedirs('LoadTest', exist_ok=True)

def main(page: Page):
    page.title = "ماژول محاسبه‌گر قابلیت اطمینان نرم‌افزار"
    page.scroll = ScrollMode.AUTO
    page.rtl = True
    page.theme_mode = ThemeMode.LIGHT

    # ------------------ WEB SYSTEM TEST ------------------

    web_growth_method_tabs = Tabs(tabs=[
        Tab(text="اجرای آزمون‌ها", content=webSys.build_tab_growth_model_run_tests(page)),
        Tab(text="محاسبه قابلیت اطمینان", content=webSys.build_tab_growth_reliability(page)),
    ])

    web_test_and_estimation_method_tabs = Tabs(tabs=[
        Tab(text="محاسبه زمان مورد نیاز برای آزمون", content=webSys.build_tab_test_and_estimation_calculate_test_time(page)),
        Tab(text="اجرای آزمون‌ها و محاسبه قابلیت اطمینان", content=webSys.build_tab_test_and_estimation_model_run_tests(page)),
        Tab(text="اصلاح دستی نتایج", content=webSys.build_tab_test_and_estimation_modify_results(page)),
    ])

    web_system_test_tabs = Tabs(tabs=[
        Tab(text="مدیریت آزمون‌ها", content=webSys.build_tab_manage_tests(page)),
        Tab(text="محاسبه قابلیت اطمینان با مدل‌های رشد", content=web_growth_method_tabs),
        Tab(text="محاسبه قابلیت اطمینان با مدل تست و تخمین", content=web_test_and_estimation_method_tabs)
    ])

    # ------------------ DESKTOP SYSTEM TEST ------------------

    desktop_growth_method_tabs = Tabs(tabs=[
        Tab(text="اجرای آزمون‌ها", content=deskSys.build_tab_growth_model_run_tests(page)),
        Tab(text="محاسبه قابلیت اطمینان", content=deskSys.build_tab_growth_reliability(page)),
    ])

    desktop_test_and_estimation_method_tabs = Tabs(tabs=[
        Tab(text="محاسبه زمان مورد نیاز برای آزمون", content=deskSys.build_tab_test_and_estimation_calculate_test_time(page)),
        Tab(text="اجرای آزمون‌ها و محاسبه قابلیت اطمینان", content=deskSys.build_tab_test_and_estimation_model_run_tests(page)),
        Tab(text="اصلاح دستی نتایج", content=deskSys.build_tab_test_and_estimation_modify_results(page)),
    ])

    desktop_system_test_tabs = Tabs(tabs=[
        Tab(text="مدیریت آزمون‌ها", content=deskSys.build_tab_manage_tests(page)),
        Tab(text="محاسبه قابلیت اطمینان با مدل‌های رشد", content=desktop_growth_method_tabs),
        Tab(text="محاسبه قابلیت اطمینان با مدل تست و تخمین", content=desktop_test_and_estimation_method_tabs)
    ])

    system_test_tabs = Tabs(tabs=[
        Tab(text="تست نرم‌افزار تحت وب", content=web_system_test_tabs),
        Tab(text="تست نرم‌افزار دسکتاپ", content=desktop_system_test_tabs)
    ])

    # ------------------ LOAD TEST ------------------

    load_test_tabs = Tabs(tabs=[
        Tab(text="تست نرم‌افزار تحت وب", content=load.build_tab_web_load_test_and_estimation(page)),
    ])

    page.add(Tabs(tabs=[
        Tab(text="مدیریت پروژه‌ها", content=pm.build_tab_project_management(page)),
        Tab(text="تست سیستم", content=system_test_tabs),
        Tab(text="تست بار و استرس", content=load_test_tabs),
    ]))

app(target=main, view=WEB_BROWSER)