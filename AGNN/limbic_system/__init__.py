"""Limbic system package - emotional modulation + confidence.

Exports cingulate gyrus. The amygdala and parahippocampal gyrus stubs
were removed (see AGNN/docs/dead-code-audit.md §2.1).
"""

from .cingulate_gyrus import CingulateGyrus

__all__ = ["CingulateGyrus"]
