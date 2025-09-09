from functools import partial

from itables import init_notebook_mode, show

init_notebook_mode(all_interactive=False)

show = partial(
    show,
    classes="compact",
    autoWidth=False,
    layout={"top1": "searchBuilder"},
    stateSave=True,
    buttons=[
        'columnsToggle',
        {
            'extend': 'colvisGroup',
            'text': 'Hide all',
            'hide': ':visible'
        },
        {
            'extend': 'colvisGroup',
            'text': 'Show all',
            'show': ':hidden'
        }
    ],
    maxBytes="1MB",
    pageLength=30
)
