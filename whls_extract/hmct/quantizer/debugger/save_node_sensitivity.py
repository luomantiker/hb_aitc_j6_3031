def save_node_sensitivity(node_info, save_dir, title=""):
    max_node_name_length = 50
    key_formatter = {}
    key_formatter["Node"] = 4
    for node, info in node_info.items():
        key_formatter["Node"] = max(len(node), key_formatter["Node"])
        for key, val in info.items():
            # Update the key format.
            if key == "Threshold":
                key_len = max(len(key), len(str(val[0])))
            else:
                key_len = max(len(key), len(str(val)))
            key_formatter[key] = (
                max(key_len, key_formatter[key]) if key in key_formatter else key_len
            )
    # Calculate the header length, and update the value in key_formatter.
    key_formatter["Node"] = min(max_node_name_length, key_formatter["Node"])
    head_length = 0
    for key in key_formatter:
        key_len = key_formatter[key] + 2
        head_length += key_len
        key_formatter[key] = "{:<" + str(key_len) + "}"

    # Generate result table.
    head = ""
    for key in key_formatter:
        head += key_formatter[key].format(key)

    equal_char = int((head_length - len(title)) / 2)
    title = "=" * equal_char + title + "=" * equal_char
    s = [title, head, "-" * head_length]
    with open(save_dir, "w") as f:
        for t in s:
            f.write(t + "\n")
        for node, info in node_info.items():
            if len(node) > max_node_name_length:
                each_log = key_formatter["Node"].format(
                    "..." + node[len(node) - max_node_name_length + 3 :],
                )
            else:
                each_log = key_formatter["Node"].format(node)

            for key, formatter in key_formatter.items():
                if key != "Node":
                    if key in info:
                        if key == "Threshold":
                            each_log += formatter.format(info[key][0])
                        else:
                            each_log += formatter.format(info[key])
                    else:
                        each_log += formatter.format("")
            f.write(each_log + "\n")
    f.close()
