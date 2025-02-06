import os


def _get_file_path(filename: str):
    cur_dir = os.path.dirname(__file__)
    file_path = os.path.realpath(os.path.join(cur_dir, filename))
    return file_path


def get_disas_path():
    return _get_file_path("bin/hbrt4-disas")


def get_run_model_nash_path():
    return _get_file_path("bin/hbrt4-run-model-nash")


def get_run_model_bayes_path():
    return _get_file_path("bin/hbrt4-run-model-bayes")


def get_libhbrt_path():
    return _get_file_path("lib/libhbrt4.so")


def get_libhbtl_path():
    return _get_file_path("lib/libhbtl.so")


def get_libnumba_path():
    return _get_file_path("lib/libhbdk_numba.so")
