# This is a Python script used to generate perf html in public release mode

import os
import json
from typing import Dict, List
from hbdk4.compiler.tools.perf import to_escaped_str, make_html_table


def gen_html_chart_of_line_common() -> str:
    return """
var option = {
  tooltip: {
    trigger: 'axis',
    axisPointer: { type: 'cross' },
  },
  title: {
    text: 'to_be_filled',
    left: 'center'
  },
  legend: {
    top: '3%%',
    data: 'to_be_filled'
  },
  xAxis: {
    type: 'category',
    boundaryGap: false,
    axisPointer: {
      type: 'shadow',
      label: { formatter: 'to_be_filled' }
    },
    axisLabel: {
      formatter: function (val) {
          return Number(val).toLocaleString() + ' us';
      }
    },
    data: 'to_be_filled',
  },
  yAxis: [
    {
      type: 'value',
      name: 'percentage',
      axisPointer: { label: {formatter: '{value}%'} },
      axisLabel: { formatter: '{value}%' }
    },
  ],
  series: 'to_be_filled'
};
"""


def round_to_keep_n_non_zero_decimal(num, n=1):
    assert num >= 0 and n >= 1
    import math

    if num == 0:
        return 0
    scale = int(-math.floor(math.log10(num)))
    if scale <= 0:
        scale = 1
    factor = 10 ** (scale + (n - 1))
    return math.floor(num * factor) / factor


def gen_html_table(total_macs, fps, ddr_read, ddr_write) -> str:
    """Generate html table contains MACs, FPS and DDR BW (read & write)"""
    div = ""
    table_setup = [
        ["Metric", "Value"],
        ["Total MACs", f"{total_macs}"],
        ["FPS", f"{round_to_keep_n_non_zero_decimal(fps, n=2)}"],
        [
            "DDR read bytes per frame",
            f"{ddr_read} bytes",
        ],
        [
            "DDR write bytes per frame",
            f"{ddr_write} bytes",
        ],
        [
            "DDR read bytes per second",
            f"{round_to_keep_n_non_zero_decimal((ddr_read * fps) / 1e9, n=2)} GB/s",
        ],
        [
            "DDR write bytes per second",
            f"{round_to_keep_n_non_zero_decimal((ddr_write * fps) / 1e9, n=2)} GB/s",
        ],
    ]
    div += '\n<h3 style="text-align: center;"> Perf Info Summary </h3>\n'
    div += make_html_table(table_setup, False)
    return div


def gen_html_chart_of_util(
    serial_id: int,
    category_interval_values: Dict[str, List[int]],
    interval_cycles: int,
    interval_ms: float,
    interval_num: int,
) -> str:
    interval_names = [
        round_to_keep_n_non_zero_decimal(interval_ms * i, n=2)
        for i in range(interval_num + 1)
    ]
    code = f"\n\nvar data_{serial_id} = [\n"
    # convert to percentage
    for category, interval_values in category_interval_values.items():
        interval_values = [int(x * 100 / interval_cycles) for x in interval_values]

        s = "{ "
        s += f'name: "{to_escaped_str(category)}", '
        s += 'type: "line", '
        s += "smooth: true, "
        s += "data: " + repr(interval_values)
        s += "}"
        code += "  %s,\n" % s
    code += "];\n"

    code += """
option.title.text = "{stage} Hardware Usage";
option.legend.data = {category_names};
option.xAxis.data = {interval_names};
option.xAxis.axisPointer.label.formatter = function (params) {{
  return Number(params.value).toLocaleString() + ' ~ '
    + parseFloat(Number(params.value) + {interval_ms}).toFixed(2).toLocaleString() + ' us';
}};
option.yAxis[0].name = "usage";
option.series = data_{serial_id};
var dom_{serial_id} = document.getElementById("util_chart_{serial_id}");
var myChart_{serial_id} = echarts.init(dom_{serial_id});
if (option && typeof option === 'object') {{
    myChart_{serial_id}.setOption(option);
}}

    """.format(
        serial_id=serial_id,
        stage="Tiling Estimation",
        category_names=repr(list(category_interval_values.keys())),
        interval_names=repr(interval_names),
        interval_ms=interval_ms,
    )

    return code


def gen_html(json_path: str):
    serial_id = 0
    html_template = """
<!DOCTYPE html>
<html style="height: 100%%">
    <head>
        <meta charset="utf-8">
        <style>
        table, th, td {
            border: 1px solid black;
            border-collapse: collapse;
            padding: 2px;
            vertical-align: center;
            text-align: center;
            margin-left: auto;
            margin-right: auto;
        }

        .hbdktooltip {
            background-color: lightyellow;
            text-align: center;
            border-radius: 6px;
            padding: 0 4px;
            border: 1px;
            border-style: solid;
            border-color: silver;
        }
        </style>
    </head>
    <body style="height: 100%%; margin: 0">
%(divs)s

        <script type="text/javascript">
        %(echart_script)s
        </script>

        <script type="text/javascript">
            %(util_chart_common_script)s
        %(util_chart_all_ss_script)s
        </script>

    </body>
</html>
"""

    with open(json_path, "r") as f:
        perf_jsons = json.load(f)

    # Common html setups for all functions inside this module
    d = dict()
    with open(os.path.join(os.path.dirname(__file__), "echarts.min.js")) as f:
        d["echart_script"] = f.read()
    d["util_chart_common_script"] = gen_html_chart_of_line_common()
    d["divs"] = ""
    d["util_chart_all_ss_script"] = ""

    for perf_json in perf_jsons:
        # It's possible that there exists many functions in a single module, we will put everything in a single html file
        for key in [
            "utilization line plot",
            "MACs",
            "FPS",
            "ddr read",
            "ddr write",
            "func name",
            "signature",
        ]:
            assert key in perf_json, f"The json file must contain field '{key}'"

        # During codegen, multiple blocks may be generated, we will skip these generated blocks
        # The signature of blocks are their ids
        if perf_json["signature"].isdigit():
            continue

        category_interval_values = {}
        interval_cycles = 0
        interval_ms = 0
        interval_num = 0
        for k, v in perf_json["utilization line plot"].items():
            if k == "interval cycles":
                interval_cycles = v
            elif k == "interval num":
                interval_num = v
            elif k == "interval us":
                interval_ms = v
            else:
                category_interval_values[k] = v

        total_macs = perf_json["MACs"]
        fps = perf_json["FPS"]
        ddr_read = perf_json["ddr read"]
        ddr_write = perf_json["ddr write"]
        func_name = perf_json["func name"]

        d["divs"] += f'\n<h1 style="text-align: center;"> {func_name} </h1>\n'
        d[
            "divs"
        ] += f"\n<div> {gen_html_table(total_macs, fps, ddr_read, ddr_write)} </div>\n"
        d[
            "divs"
        ] += f'\n        <div id="util_chart_{serial_id}" style="height: 60%;"> </div>'
        d["util_chart_all_ss_script"] += gen_html_chart_of_util(
            serial_id,
            category_interval_values,
            interval_cycles,
            interval_ms,
            interval_num,
        )
        serial_id += 1

    parent_dir = os.path.dirname(json_path)
    html_filename = os.path.join(parent_dir, "public_perf.html")
    print(f"Perf html generated to {html_filename}")
    with open(html_filename, "w") as f:
        f.write(html_template % d)


if __name__ == "__main__":
    print("this script should not be invoked")
