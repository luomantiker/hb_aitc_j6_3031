from .version import __version__


def set_library_path():
    import os

    ld_library_path = os.getenv("LD_LIBRARY_PATH")
    link_dir = os.path.dirname(os.path.abspath(__file__))
    link_dir_list = [
        os.path.join(link_dir, "compiler"),
        os.path.join(link_dir, "ir/horizon_onnx"),
        os.path.join(link_dir, "horizon_onnxruntime"),
    ]
    if ld_library_path is not None:
        link_dir_list.append(ld_library_path)
    ld_library_path = ":".join(link_dir_list)
    os.environ["LD_LIBRARY_PATH"] = ld_library_path


set_library_path()
