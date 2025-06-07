from contextlib import contextmanager
from typing import Any

_ms_comm_context: Any = None


class MultiStreamContext:

    def __init__(self):
        self.cur_micro_batch_num: int = -1
        self.ms_layer_index_context: int = -1
        self.ms_metadata_context: Any = None
        self.ms_attn_metadata_context: Any = None

    def set_multistream_layer_context(self, start_layer: int, ms_metadata: Any,
                                      attn_metadata: Any):
        """
        set multistream layer context before transformer layers
        """
        #global _ms_layer_index_context, _ms_metadata_context, _ms_attn_metadata_context
        self.ms_layer_index_context = start_layer
        self.ms_metadata_context = ms_metadata
        self.ms_attn_metadata_context = attn_metadata

    def reset_multistream_layer_context(self):
        """
        reset multistream layer context
        """
        #global _ms_layer_index_context, _ms_metadata_context, _ms_attn_metadata_context
        self.ms_layer_index_context = -1
        self.ms_metadata_context = None
        self.ms_attn_metadata_context = None

    def get_multistream_layer_context(self):
        """
        get multistream layer context
        """
        return self.ms_layer_index_context, self.ms_metadata_context, self.ms_attn_metadata_context

    def advance_step_multistream_layer_context(self):
        """
        advance multistream layer index context
        """
        #global _ms_layer_index_context
        self.ms_layer_index_context += 1

    def get_multistream_microbatch_context(self) -> int:
        return self.cur_micro_batch_num


def get_multistream_comm_context() -> Any:
    """Get the current comm forward context."""
    return _ms_comm_context


@contextmanager
def set_multistream_context(context: Any):
    """A context manager that stores the current comm forward context,
    can be attention metadata, etc."""
    global _ms_comm_context
    _ms_comm_context = context
    try:
        yield
    finally:
        _ms_comm_context = None
