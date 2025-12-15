from flet import *

def build_tab_project_management(page: Page):
    all_projects = ["Test" for i in range(5)]

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
        all_projects.pop(name, None)
        # save_all_testcases(all_projects)
        user_message.value = f"پروژه '{name}' حذف شد."
        refresh_project_list()

    def save_project(e):
        new_name = new_project_name_input.value
        if not new_name:
            user_message.value = "لطفاً نام پروژه را وارد کنید."
            user_message.color = Colors.RED
            page.update()
            return
        
        # save_all_testcases(all_testcases)
        user_message.value = f"سناریو '{new_name}' ذخیره شد."
        user_message.color = Colors.GREEN
        refresh_project_list()

    def on_project_select(e):
        name = selected_project.value
        if name and name in all_projects:
            data = all_projects[name]
            new_project_name_input.value = name
            new_project_name_input.read_only = True
        else:
            new_project_name_input.value = ""
            new_project_name_input.read_only = False
        page.update()
    
    def select_project(e):
        pass

    selected_project.on_change = on_project_select

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
