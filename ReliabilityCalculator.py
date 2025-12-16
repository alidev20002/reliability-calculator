import os
from flet import *
import WebSystemTestTabs as webSys
import DesktopSystemTestTabs as deskSys
import WebLoadTestTabs as webLoad
import ProjectManagementTab as pm

os.makedirs('web', exist_ok=True)
os.makedirs('desktop', exist_ok=True)

def main(page: Page):
    page.title = "ماژول محاسبه‌گر قابلیت اطمینان نرم‌افزار"
    page.scroll = ScrollMode.AUTO
    page.rtl = True
    page.theme_mode = ThemeMode.LIGHT

    # ------------------ WEB PLATFORM ------------------

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

    web_load_test_tabs = Tabs(tabs=[
        Tab(text="تست نرم‌افزار تحت وب", content=webLoad.build_tab_web_load_test_and_estimation(page)),
    ])

    web_tabs = Tabs(tabs=[
        Tab(text="تست سیستم (Katalon Recorder)", content=web_system_test_tabs),
        Tab(text="تست بار و استرس (Apache Jmeter)", content=web_load_test_tabs)
    ])

    # ------------------ DESKTOP PLATFORM ------------------

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

    desktop_tabs = Tabs(tabs=[
        Tab(text="تست سیستم (Sikulix)", content=desktop_system_test_tabs),
    ])

    # ------------------ Putting All Together ------------------

    page.add(Tabs(tabs=[
        Tab(text="مدیریت پروژه‌ها", content=pm.build_tab_project_management(page)),
        Tab(text="نرم‌افزار تحت وب", content=web_tabs),
        # Tab(text="نرم‌افزار دسکتاپ", content=desktop_tabs),
    ]))

app(target=main, view=WEB_BROWSER)