# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)


class DFlashGenerationCancelled(RuntimeError):
    """Raised when a request asks DFlash generation to stop at a safe boundary."""
