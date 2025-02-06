# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging

from . import onnxruntime_pybind11_state as onnxruntime_cpp2py_export
from .onnxruntime_pybind11_state import get_available_providers


class InferenceSession:
    """This is the main class used to run a model."""

    def __init__(self, path_or_bytes, sess_options=None, providers=None):
        """Initialize function for InferenceSession.

        Args:
            path_or_bytes: filename or serialized model in a byte string
            sess_options: session options
            providers: providers to use for session. If empty, will use
                all available providers.
        """
        providers = [] if providers is None else providers

        self._path_or_bytes = path_or_bytes
        self._sess_options = sess_options
        self._load_model(providers)
        self._enable_fallback = True

    def _load_model(self, providers=None):
        if isinstance(self._path_or_bytes, str):
            self._sess = onnxruntime_cpp2py_export.InferenceSession(
                self._sess_options
                or onnxruntime_cpp2py_export.get_default_session_options(),
                self._path_or_bytes,
                True,
            )
        elif isinstance(self._path_or_bytes, bytes):
            self._sess = onnxruntime_cpp2py_export.InferenceSession(
                self._sess_options
                or onnxruntime_cpp2py_export.get_default_session_options(),
                self._path_or_bytes,
                False,
            )
        # elif isinstance(self._path_or_bytes, tuple):
        # to remove, hidden trick
        #   self._sess.load_model_no_init(self._path_or_bytes[0], providers)
        else:
            raise TypeError(f"Unable to load from type '{type(self._path_or_bytes)}'")

        providers = [] if providers is None else providers
        self._sess.load_model(providers)

        self._session_options = self._sess.session_options
        self._inputs_meta = self._sess.inputs_meta
        self._outputs_meta = self._sess.outputs_meta
        self._overridable_initializers = self._sess.overridable_initializers
        self._model_meta = self._sess.model_meta
        self._providers = self._sess.get_providers()

        # Tensorrt can fall back to CUDA. All others fall back to CPU.
        if (
            "TensorrtExecutionProvider"
            in onnxruntime_cpp2py_export.get_available_providers()
        ):
            self._fallback_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            self._fallback_providers = ["CPUExecutionProvider"]

    def _reset_session(self):
        """Release underlying session object."""
        # meta data references session internal structures
        # so they must be set to None to decrement _sess reference count.
        self._inputs_meta = None
        self._outputs_meta = None
        self._overridable_initializers = None
        self._model_meta = None
        self._providers = None
        self._sess = None

    def get_session_options(self):
        """Return the session options. See :class:`onnxruntime.SessionOptions`."""
        return self._session_options

    def get_inputs(self):
        """Return the inputs metadata as a list of :class:`onnxruntime.NodeArg`."""
        return self._inputs_meta

    def get_outputs(self):
        """Return the outputs metadata as a list of :class:`onnxruntime.NodeArg`."""
        return self._outputs_meta

    def get_overridable_initializers(self):
        """Return the inputs (including initializers) metadata as a list of :class:`onnxruntime.NodeArg`."""  # noqa: E501
        return self._overridable_initializers

    def get_modelmeta(self):
        """Return the metadata. See :class:`onnxruntime.ModelMetadata`."""
        return self._model_meta

    def get_providers(self):
        """Return list of registered execution providers."""
        return self._providers

    def set_providers(self, providers):
        """Register the input list of execution providers.

        The underlying session is re-created.
        The list of providers is ordered by Priority. For example
        ['CUDAExecutionProvider', 'CPUExecutionProvider'] means
        execute a node using CUDAExecutionProvider if capable,
        otherwise execute using CPUExecutionProvider.

        Args:
            providers: list of execution providers
        """
        if not set(providers).issubset(
            onnxruntime_cpp2py_export.get_available_providers()
        ):
            raise ValueError(
                f"{providers} does not contain a subset of available providers "
                f"{onnxruntime_cpp2py_export.get_available_providers()}"
            )
        self._reset_session()
        self._load_model(providers)

    def disable_fallback(self):
        """Disable session.run() fallback mechanism."""
        self._enable_fallback = False

    def enable_fallback(self):
        """Enable session.Run() fallback mechanism.

        If session.Run() fails due to an internal Execution Provider failure,
        reset the Execution Providers enabled for this session.
        If GPU is enabled, fall back to CUDAExecutionProvider.
        otherwise fall back to CPUExecutionProvider.
        """
        self._enable_fallback = True

    def run(self, output_names, input_feed, run_options=None):
        """Compute the predictions.

        Args:
            output_names: name of the outputs
            input_feed: dictionary ``{ input_name: input_value }``
            run_options (optional): See :class:`onnxruntime.RunOptions`.

        Examples:
            >>> sess.run([output_name], {input_name: x})
        """
        num_required_inputs = len(self._inputs_meta)
        num_inputs = len(input_feed)
        # the graph may have optional inputs used to override initializers.
        # allow for that.
        if num_inputs < num_required_inputs:
            raise ValueError(
                f"Model requires {num_required_inputs} inputs. "
                f"Input Feed contains {num_inputs}"
            )
        if not output_names:
            output_names = [output.name for output in self._outputs_meta]
        try:
            return self._sess.run(output_names, input_feed, run_options)
        except onnxruntime_cpp2py_export.EPFail as err:
            if self._enable_fallback:
                logging.info(f"EP Error: {err!s} using {self._providers}")
                logging.info(
                    f"Falling back to {self._fallback_providers} and retrying."
                )
                self.set_providers(self._fallback_providers)
                # Fallback only once.
                self.disable_fallback()
                return self._sess.run(output_names, input_feed, run_options)
            raise

    def end_profiling(self):
        """End profiling and return results in a file.

        The results are stored in a filename if the option
        :meth:`onnxruntime.SessionOptions.enable_profiling`.
        """
        return self._sess.end_profiling()
