"""Pin matplotlib's config/cache dir to local temp space -- import BEFORE matplotlib.

The default MPLCONFIGDIR (``~/.cache/matplotlib``) may sit on a network home directory,
where ``usetex``'s temporary tex cache can fail to clean up (``OSError: [Errno 39]
Directory not empty``). ``setdefault`` keeps any user-provided ``MPLCONFIGDIR``.
"""

import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), f"mplconfig-{os.getuid()}")
)
