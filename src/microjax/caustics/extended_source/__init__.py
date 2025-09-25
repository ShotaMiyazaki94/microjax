# This file is a modified and extended version of code from the `caustics` package:
#   https://github.com/fbartolic/caustics
# Originally developed by Fran Bartolic under the MIT License.
#
# modifications and extensions have been made by Shota Miyazaki for the `microjax` project.
#
# SPDX-FileCopyrightText: 2022 Fran Bartolic
# SPDX-FileCopyrightText: 2025 Shota Miyazaki
# SPDX-License-Identifier: MIT

"""Extended-source magnification helpers for microJAX.

This subpackage breaks the original monolithic module into smaller pieces while
preserving the public API (``mag_extended_source``).
"""

from .mag import mag_extended_source

__all__ = ["mag_extended_source"]
