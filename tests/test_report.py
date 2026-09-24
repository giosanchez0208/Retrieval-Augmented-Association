from reidtrack.report import format_table


def test_table_has_rules_alignment_caption_footer_and_note():
    text = format_table(
        ["name", "value"],
        [["a", 1], ["bbb", 22]],
        caption="Table",
        footer=[["total", 23]],
        note="values in ms",
    )

    assert text.splitlines() == [
        "Table",
        "",
        "------------",
        "name   value",
        "------------",
        "a          1",
        "bbb       22",
        "------------",
        "total     23",
        "------------",
        "values in ms",
    ]
