"""Avoid per-header Lustre stats when Python 3.12 infers top-level package names."""

import collections
import csv
import functools
import importlib.metadata as metadata
import inspect
import sys


ORIGINAL_PACKAGES_DISTRIBUTIONS = getattr(
    metadata.packages_distributions, "__wrapped__", metadata.packages_distributions
)


def _inferred_names(dist):
    # Preserve custom distribution providers and legacy egg-info behavior.
    if type(dist) is not metadata.PathDistribution:
        return metadata._top_level_inferred(dist)
    record = dist.read_text("RECORD")
    if not record:
        return metadata._top_level_inferred(dist)

    names = set()
    try:
        for row in csv.reader(record.splitlines()):
            if not 1 <= len(row) <= 3:
                return metadata._top_level_inferred(dist)
            # Keep the stdlib's malformed-metadata errors rather than hiding them.
            if len(row) > 1 and row[1]:
                metadata.FileHash(row[1])
            if len(row) > 2 and row[2]:
                int(row[2])
            path = metadata.PackagePath(row[0])
            name = path.parts[0] if len(path.parts) > 1 else inspect.getmodulename(path)
            if name is None or "." in name or name in names:
                continue
            # The stdlib checks every RECORD entry before building a set of names.
            # The set needs only one existing file per name. Missing first entries
            # still fall through to later entries, and entirely missing names stay absent.
            if dist.locate_file(path).exists():
                names.add(name)
    except (csv.Error, TypeError, ValueError):
        return metadata._top_level_inferred(dist)
    return names


@functools.wraps(ORIGINAL_PACKAGES_DISTRIBUTIONS)
def packages_distributions():
    result = collections.defaultdict(list)
    for dist in metadata.distributions():
        for name in metadata._top_level_declared(dist) or _inferred_names(dist):
            result[name].append(dist.metadata["Name"])
    return dict(result)


def install():
    # The inference rule above matches the pinned Python 3.12 stdlib. Other
    # interpreters retain their own implementation instead of guessing its semantics.
    if sys.version_info[:2] == (3, 12):
        metadata.packages_distributions = packages_distributions
