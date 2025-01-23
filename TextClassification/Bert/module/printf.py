from rich import print
from rich.table import Table
from rich.align import Align
from rich.console import Console

def reset_console(args):
    """
    重置终端，便于打印log信息。
    """
    console = Console()
    table = Table(show_footer=False)
    table.title = ("[bold not italic]:robot:[/] Config Parameters")
    table.add_column("key", no_wrap=True)
    table.add_column("value", no_wrap=True)

    for arg in vars(args):
        table.add_row(arg, str(getattr(args, arg)))

    table.caption = "You can change config in [b not dim]Source Code[/]"
    table.columns[0].style = "bright_red"
    table.columns[0].header_style = "bold bright_red"
    table.columns[1].style = "bright_green"
    table.columns[1].header_style = "bold bright_green"
    table_centered = Align.center(table)
    console.print(table_centered)
