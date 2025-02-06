import os
import json
import functools
import http.server
import socketserver
import socket


def generate_html_template():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualization of the Dynamic Programming Procedure</title>
    <style>
        table {
            border-collapse: collapse;
        }

        th, td {
            border: 1px solid #ddd;
            padding: 1px;
            text-align: left;
        }

        .block {
            display: inline-block;
            margin: 1px;
            text-align: left;
            line-height: 15px;
            cursor: pointer;
            background-color: lightgrey; /* Default color */
            white-space: normal; /* Allow text to wrap within the block */
            word-wrap: break-word; /* Break words to fit within the block */
            font-size: 10px;
        }

        .info {
            display: none;
            position: absolute;
            background-color: #fff;
            border: 1px solid #ccc;
            padding: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            white-space: normal;
            min-width: 1000px
        }

        tr {
            border-bottom: 2px solid #ddd;
        }

        th {
            background-color: #045CF4;
            color: white;
        }

    </style>
</head>
<body>
    %(tables)s
<script>
    %(blockLayers)s
    var tables =  Array.from(document.getElementsByTagName('table'));
    tables.forEach(function(table, index) {
        var elements = Array.from(table.getElementsByClassName('block'));
        elements.forEach(function(element) {
            element.addEventListener('mouseover', function() {
            var infoDiv = element.getElementsByClassName('info')[0];
            if(infoDiv.innerText.trim() == ""){
                return;
            }
            var doingLayerIds = Array.from(infoDiv.innerText.trim().split(","));
            var displayText = infoDiv.innerText;
            doingLayerIds.forEach(function(doingLayerId){
                var key = index + "_" + doingLayerId
                if(blockLayers.has(key)){
                    displayText += "\\n" + blockLayers.get(key);
                }
            });
            infoDiv.innerText = displayText;
            infoDiv.style.display = 'block';
            });

            element.addEventListener('mouseout', function() {
            var infoDiv = element.getElementsByClassName('info');
            infoDiv[0].innerText = infoDiv[0].innerHTML.split("<br>")[0];
            infoDiv[0].style.display = 'none';
            });
        });

        var tableRows = Array.from(table.rows);
        tableRows.forEach(function(tableRow){
            var tableCells = Array.from(tableRow.cells);
            var rowIndex = tableRow.rowIndex;
            var numOfCollapsedRow = 0;

            var tableCellOfCollapsedRow = [];
            tableCells.forEach(function(cell){
                if(cell.style.boxShadow != ""){
                    numOfCollapsedRow += 1;
                    tableCellOfCollapsedRow.push(cell);
                }
            });

            tableCells.forEach(function(cell){
                if(cell.style.boxShadow != ""){
                    cell.addEventListener('click', () => {
                        var i = 0;
                        for(let j = 0; j < numOfCollapsedRow; j++){
                            if( tableCellOfCollapsedRow[j] == cell){
                                i = j;
                                continue;
                            }
                            var otherCell = tableCellOfCollapsedRow[j];
                            if(tableRows[rowIndex + j + 1].style.display !== 'none'){
                                tableRows[rowIndex + j + 1].style.display = 'none';
                                otherCell.style.border = '';
                            }
                        }

                        var collapsedRow = tableRows[rowIndex + i + 1];
                        if(collapsedRow.style.display === 'none'){
                            cell.style.border = '2px solid blue';
                            collapsedRow.style.display = 'table-row';
                        }else{
                            cell.style.border = '';
                            collapsedRow.style.display = 'none';
                        }

                    });
                }
            });
        });


    });
  </script>

</body>
</html>
"""


def generate_data_cell(content, style=""):
    data_cell = f'<td style="{style}"> {content} </td>'
    return data_cell


def generate_table_row(content, style=""):
    table_row = f'<tr style="{style}"> {content} </tr>'
    return table_row


def generate_header_html(row_entry_count):
    header_row_html = "<th>E</th>"
    header_row_html += f'<th colspan="{row_entry_count}">Status</th>'
    return header_row_html


def create_block(
    innerText,
    outerText,
    isLastBlock=True,
    textColor="black",
    width="",
    background="rgba(0, 0, 0, 0) none repeat scroll 0% 0%",
):
    html_template = """
    <div class="block" style="%(outerStyle)s">
    %(outerText)s
    <div class="info" style="%(innerStyle)s"> %(innerText)s </div>
    </div>
    """
    d = {"outerText": outerText, "innerText": innerText}
    d["outerStyle"] = f"background: {background}; color: {textColor};"
    if width != "":
        d["outerStyle"] += f" width={width};"
    d["innerStyle"] = f"color: {textColor};"

    content = html_template % d
    if isLastBlock:
        return content
    return content + "<br>"


def create_entry(entryContents, color, textColor, boxShadow=""):
    cell_content = ""
    cell_style = f"background-color: {color}; box-shadow: {boxShadow}"
    numOfentryContents = len(entryContents)
    for i in range(numOfentryContents):
        content = entryContents[i]
        block = create_block(
            content["innerText"],
            content["outerText"],
            i == numOfentryContents - 1,
            textColor,
            "100px",
        )
        cell_content += block
    return generate_data_cell(cell_content, cell_style)


def create_row(headerContent, entryContents, colors, textColors):
    rowContent = ""
    headerCell = generate_data_cell(
        create_block(headerContent["innerText"], headerContent["outerText"], True)
    )

    shadowValue = "inset 1px 1px 1px #888888"
    numNonCollapsedEntries = len(entryContents)
    faceRowContent = headerCell
    for i in range(numNonCollapsedEntries):
        numCollapsedEntries = len(entryContents[i])
        boxShadow = ""
        if numCollapsedEntries > 1:
            boxShadow = shadowValue
        faceEntry = create_entry(
            entryContents[i][0], colors[i][0], textColors[i][0], boxShadow
        )
        faceRowContent += faceEntry
    rowContent += generate_table_row(faceRowContent)

    collapsedRowContent = ""
    for i in range(numNonCollapsedEntries):
        numCollapsedEntries = len(entryContents[i])
        if numCollapsedEntries > 1:
            collapsedRow = ""
            headerCellOfCollapsedRow = generate_data_cell(
                create_block("", "", background="#045CF4")
            )
            collapsedRow += headerCellOfCollapsedRow
            for j in range(numCollapsedEntries):
                entry = create_entry(
                    entryContents[i][j], colors[i][j], textColors[i][j]
                )
                collapsedRow += entry
            collapsedRowStyle = "display: none"
            collapsedRowContent += generate_table_row(collapsedRow, collapsedRowStyle)
    rowContent += collapsedRowContent
    return rowContent


def generate_table_entries(block_info):
    def discrete_list_to_ranges(input_list):
        """Given sorted list of unique integers, this function groups consecutive numbers in ranges and display them in the form of '[1~10], [11], [20~25]'

        Args:
            input_list (List[int]): a list of sorted unique integers

        Returns:
            str: string form of ranges in the given input_list
        """
        result = []
        start = input_list[0]

        for i in range(1, len(input_list)):
            if input_list[i] != input_list[i - 1] + 1:
                if start == input_list[i - 1]:
                    result.append(str(start))
                else:
                    result.append(f"{start}~{input_list[i - 1]}")

                start = input_list[i]

        if start == input_list[-1]:
            result.append(str(start))
        else:
            result.append(f"{start}~{input_list[-1]}")

        result_str = "[" + "],[".join(result) + "]"
        return result_str

    def get_num_of_computing_layers(layer_id_to_layer, layer_ids):
        # hbdk.view and hbdk.broadcast and hbir.constant are not computing ops
        computing_layers_cnt = 0
        for layer_id in layer_ids:
            layer_name = layer_id_to_layer[layer_id]
            if (
                ("hbdk.view" not in layer_name)
                and ("hbdk.broadcast" not in layer_name)
                and ("hbir.constant" not in layer_name)
            ):
                computing_layers_cnt += 1

        return computing_layers_cnt

    layer_id_to_layer = {
        int(id): layer for id, layer in block_info["blockLayers"].items()
    }

    table_content = ""

    func_signature_id = -1
    for row_idx, status in enumerate(block_info["statuses"]):
        d = {}
        # Each status is a JSON object, representing all statuses explored during DP that end with one specific layer

        ending_layer_id = status["endingLayerId"]
        headerContent = {"innerText": ending_layer_id, "outerText": ending_layer_id}

        entry_contents = {}
        for status_info in status["statusInfos"]:
            entry_content = []
            # Each status_info is a JSON object, representing one status, and will be displayed in one table entry in the generated html file
            doing_layer_ids = sorted(status_info["doingLayerIds"])
            computing_layers_cnt = get_num_of_computing_layers(
                layer_id_to_layer, doing_layer_ids
            )
            total_layers_cnt = len(doing_layer_ids)
            if computing_layers_cnt not in entry_contents:
                entry_contents[computing_layers_cnt] = []

            is_pruned = status_info["isPruned"]
            is_selected = status_info["isSelected"]
            l1m_tensors = []
            if "l1m_tensors" in status_info["detailInfo"]:
                l1m_tensors = status_info["detailInfo"][
                    "l1m_tensors"
                ]  # l1m_tensors is a List of JSON object

            doing_layer_ids_str = discrete_list_to_ranges(doing_layer_ids)

            entry_content.append(
                {
                    "innerText": ",".join([str(_) for _ in doing_layer_ids]),
                    "outerText": f"doing_layers: {doing_layer_ids_str}",
                }
            )

            # Prepare l1m tensors
            l1m_tensors_list = []
            l1m_tensors_outer_text_list = []
            for l1m_tensor_info in l1m_tensors:
                defining_op_id = l1m_tensor_info["op_id"]
                l1m_tensors_list.append(
                    defining_op_id if (defining_op_id >= 0) else func_signature_id
                )
                l1m_tensor_outer_text_list = []
                for k, v in l1m_tensor_info.items():
                    l1m_tensor_outer_text_list.append(
                        f"{k}: {v:,.0f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                    )
                l1m_tensors_outer_text_list.append(
                    ", ".join(l1m_tensor_outer_text_list)
                )
            entry_content.append(
                {
                    "innerText": ",".join([str(_) for _ in l1m_tensors_list]),
                    "outerText": f"l1m_tensors: {l1m_tensors_outer_text_list}",
                }
            )

            # Prepare other entries in "detailInfo"
            for k, v in status_info["detailInfo"].items():
                if k == "l1m_tensors":
                    continue
                entry_content.append(
                    {
                        "innerText": "",
                        "outerText": f"{k}: {v:,.0f}"
                        if isinstance(v, (int, float))
                        else f"{k}: {v}",
                    }
                )

            # Prepare colors
            background_color_pruned = "lightgrey"
            background_color_selected = "lightgreen"
            background_color_others = "white"
            text_color_pruned = "grey"
            text_color_selected = "black"
            text_color_others = "black"
            assert not (
                is_pruned and is_selected
            ), "Can't be selected and pruned at the same time"
            background_color = background_color_others
            text_color = text_color_others
            if is_pruned:
                background_color = background_color_pruned
                text_color = text_color_pruned
            elif is_selected:
                background_color = background_color_selected
                text_color = text_color_selected

            # Update entry contents
            color_key = 2
            if background_color == background_color_others:
                color_key = 1
            elif background_color == background_color_selected:
                color_key = 0
            entry_contents[computing_layers_cnt].append(
                (
                    color_key,
                    total_layers_cnt,
                    entry_content,
                    background_color,
                    text_color,
                )
            )

        # Group entries that have the same number of computing ops together
        grouped_entry_contents = []
        grouped_entry_background_colors = []
        grouped_entry_text_colors = []
        for computing_ops_cnt in sorted(entry_contents.keys()):
            entry_contents[computing_ops_cnt].sort(key=lambda x: x[0])
            contents = []
            background_colors = []
            text_colors = []
            for _, _, content, background_color, text_color in entry_contents[
                computing_ops_cnt
            ]:
                contents.append(content)
                background_colors.append(background_color)
                text_colors.append(text_color)

            grouped_entry_contents.append(contents)
            grouped_entry_background_colors.append(background_colors)
            grouped_entry_text_colors.append(text_colors)

        rowContent = create_row(
            headerContent,
            grouped_entry_contents,
            grouped_entry_background_colors,
            grouped_entry_text_colors,
        )
        table_content += rowContent
    return table_content


def generate_html_for_one_block(block_name, block_info):
    html_template_for_one_table = """
    <h1>%(block_name)s</h1>
    <table>
      %(header_row)s

      %(table_entries)s
    </table>
    """

    # Generate the table entries
    d = {"block_name": block_name}
    row_entry_count = functools.reduce(
        max, [len(status["statusInfos"]) for status in block_info["statuses"]]
    )
    d["header_row"] = generate_table_row(generate_header_html(row_entry_count))

    # Generate the table entries
    d["table_entries"] = generate_table_entries(block_info)

    # Replace and return
    return html_template_for_one_table % d


def sub_command_dp(subparsers):
    """as a sub-command of `hbdk-view` tool"""

    parser = subparsers.add_parser(
        "dp", help="visualize the dynamic programming's process"
    )
    parser.add_argument("json_path", type=str, help="path to the generated json file")
    parser.add_argument(
        "--no-http-server",
        action="store_true",
        help="only generate html, do not launch the http server",
    )

    def runner(args):
        if os.path.exists(args.json_path):
            with open(args.json_path) as f:
                data = json.load(f)

            html_template = generate_html_template()
            tables = ""
            block_layers = []
            for table_idx, (block_name, block_info) in enumerate(
                data["blockInfos"].items()
            ):
                tables += generate_html_for_one_block(block_name, block_info)
                func_signature = block_info["funcSignature"]
                func_signature_id = -1
                for id, layer in block_info["blockLayers"].items():
                    block_layers.append(f"['{table_idx}_{id}', `{layer}`]")
                block_layers.append(
                    f"['{table_idx}_{func_signature_id}', `{func_signature}`]"
                )
            js_template_above_all = """
            const blockLayers = new Map([
            %(block_layers)s
            ]);
            """

            d = {
                "tables": tables,
                "blockLayers": js_template_above_all
                % {"block_layers": ",".join(block_layers)},
            }

            json_filename = os.path.splitext(os.path.basename(args.json_path))[0]
            dir_name = os.path.dirname(args.json_path)
            html_filename = os.path.join(dir_name, json_filename + ".html")
            # The result html will be generated under the same directory of the given json and they will have the same name
            with open(html_filename, "w") as f:
                print(html_template % d, file=f)
                print(f"Visualization result generated to {html_filename}")

            if args.no_http_server:
                return
            # Open a HTTP server so the generated html can be viewed via the browser
            if dir_name:
                # If dir_name is empty, it means the generated html is in the current directory, so os.chdir is only needed when dir_name is not empty
                os.chdir(dir_name)

            # A custom handler to suppress log message
            class QuietHandler(http.server.SimpleHTTPRequestHandler):
                def log_message(self, format, *args):
                    pass

            # Create a socket to find an available port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", 0))
                _, port = s.getsockname()

            # Start a http server and close it upon Ctrl+C
            print(
                f"\033[92mVisit http://localhost:{port}/{json_filename + '.html'} for visualization\033[0m"
            )
            print(
                'NOTE: you may need to change "localhost" to the actual IP address or the hostname of the remote server you are currently running this command on, or you can forward the port using SSH tunneling'
            )
            http_server = socketserver.TCPServer(("", port), QuietHandler)
            try:
                http_server.serve_forever()
            except KeyboardInterrupt:
                print("Shutting down the http server ...")
                http_server.shutdown()
        else:
            print("Cannot find corresponding json file in the provided path")

    parser.set_defaults(func=runner)


if __name__ == "__main__":
    print("this script should not be invoked")
