from flet import *
import json
import os

PROJECT_CONFIG = 'project_config.json'

def load_prject_config():
    if os.path.exists(PROJECT_CONFIG):
        with open(PROJECT_CONFIG, "r", encoding="utf-8") as f:
            return json.load(f)
    return {'selected_project': 'default', 'jmeter_path': '', 'sikulix_path': '', 'projects': []}

def save_project_config(data):
    with open(PROJECT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def build_tab_project_management(page: Page):
    project_config = load_prject_config()
    all_projects = project_config['projects']

    selected_project = Dropdown(
        label="انتخاب پروژه",
        text_align='right',
        text_style=TextStyle(
            size=14
        ),
        width=300
    )

    new_project_name_input = TextField(label="نام پروژه جدید")
    project_list_column = Column(scroll=ScrollMode.AUTO, height=300)
    user_message = Text()

    def refresh_project_list():
        selected_project.options = [dropdown.Option(p) for p in all_projects]
        selected_project.value = project_config['selected_project']
        project_list_column.controls = []
        for name in all_projects:
            project_list_column.controls.append(
                ListTile(
                    title=Text(f"{name}"),
                    trailing=Icon(name="DELETE", color="red"),
                    on_click=lambda e, name=name: delete_project(name)
                )
            )
        page.update()

    def delete_project(name):
        project_config['projects'].remove(name)
        save_project_config(project_config)
        user_message.value = f"پروژه '{name}' حذف شد."
        user_message.color = Colors.GREEN
        refresh_project_list()

    def save_project(e):
        new_name = new_project_name_input.value
        if not new_name:
            user_message.value = "لطفاً نام پروژه را وارد کنید."
            user_message.color = Colors.RED
            page.update()
            return
        
        if new_name in project_config['projects']:
            user_message.value = "پروژه تکراری است."
            user_message.color = Colors.RED
            page.update()
            return
        
        project_config['projects'].append(new_name)
        save_project_config(project_config)
        user_message.value = f"پروژه '{new_name}' ذخیره شد."
        user_message.color = Colors.GREEN
        refresh_project_list()      
    
    def select_project(e):
        if selected_project.value:
            project_config['selected_project'] = selected_project.value
            save_project_config(project_config)
            user_message.value = f"پروژه '{selected_project.value}' انتخاب شد."
            user_message.color = Colors.GREEN
            page.update()

    refresh_project_list()

    return Column([
        Container(
            content=Text("مدیریت پروژه‌ها ", size=20, weight="bold", text_align="center"),
            alignment=alignment.center,
            padding=30
        ),
        Row([
            Container(
                Column([
                    Text("لیست پروژه‌ها", style=TextThemeStyle.TITLE_MEDIUM),
                    project_list_column,
                ], width=300, horizontal_alignment="center", alignment="start", spacing=50),
                bgcolor="#dfdfdf",
                padding=30
            ),
            
            Column([
                Column([
                    new_project_name_input,
                    ElevatedButton(
                        text="ذخیره پروژه",
                        bgcolor=Colors.BLUE_500,
                        color=Colors.WHITE,
                        style=ButtonStyle(
                            shape= RoundedRectangleBorder(8),
                            padding=Padding(15, 15, 15, 15)
                        ),
                        on_click=save_project
                    ),
                ], horizontal_alignment='center'),
                Column([
                    selected_project,
                    ElevatedButton(
                        text="انتخاب پروژه",
                        bgcolor=Colors.BLUE_500,
                        color=Colors.WHITE,
                        style=ButtonStyle(
                            shape= RoundedRectangleBorder(8),
                            padding=Padding(15, 15, 15, 15)
                        ),
                        on_click=select_project
                    ),
                ], horizontal_alignment='center'),
                user_message
            ], horizontal_alignment="start", spacing=50)
        ], alignment='start', vertical_alignment='start', expand=True, spacing=50),
    ], spacing=50)
