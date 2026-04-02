"""Filesystem tools for the agent: ls, read_file, glob, grep, write_file, edit_file."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Input schemas                                                                #
# --------------------------------------------------------------------------- #

class ReadFileInput(BaseModel):
    file_path: str = Field(description="Absolute path to the file")
    offset: int = Field(default=0, description="Starting line (0-indexed)")
    limit: int = Field(default=150, description="Max lines to read")


class WriteFileInput(BaseModel):
    file_path: str = Field(description="Absolute path to write")
    content: str = Field(description="Full content to write")


class EditFileInput(BaseModel):
    file_path: str = Field(description="Absolute path to the file")
    old_string: str = Field(description="Exact string to find and replace (must be unique in file)")
    new_string: str = Field(description="Replacement string")


class LsInput(BaseModel):
    path: str = Field(description="Directory path to list")


class GlobInput(BaseModel):
    pattern: str = Field(description="Glob pattern (e.g. '**/*.py')")
    root: str = Field(description="Root directory to search from")


class GrepInput(BaseModel):
    pattern: str = Field(description="Regex pattern to search")
    path: str = Field(description="File or directory path to search in")
    glob: str = Field(default="*", description="File glob filter (e.g. '*.py')")


# --------------------------------------------------------------------------- #
# Tools                                                                        #
# --------------------------------------------------------------------------- #

@tool(args_schema=LsInput)
def ls(path: str) -> str:
    """List files and directories at the given path."""
    target = Path(path)
    if not target.exists():
        return f"Error: path_not_found: {path}"
    if not target.is_dir():
        return f"Error: not_a_directory: {path}"

    entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
    lines = []
    for entry in entries:
        prefix = "" if entry.is_dir() else "  "
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"{prefix}{entry.name}{suffix}")
    return "\n".join(lines) if lines else "(empty directory)"


@tool(args_schema=ReadFileInput)
def read_file(file_path: str, offset: int = 0, limit: int = 150) -> str:
    """Read file content with line numbers (cat -n format).

    Returns content with 1-based line numbers. Includes truncation hint if needed.
    """
    path = Path(file_path)
    if not path.exists():
        return f"Error: file_not_found: {file_path}"
    if path.is_dir():
        return f"Error: is_directory: {file_path}"

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        return f"Error reading file: {e}"

    selected = lines[offset: offset + limit]
    formatted = "\n".join(
        f"{offset + i + 1:>6}\t{line}" for i, line in enumerate(selected)
    )

    if offset + limit < len(lines):
        formatted += (
            f"\n\n[Output truncated at line {offset + limit}. "
            f"Use offset={offset + limit} to read more.]"
        )

    return formatted


@tool(args_schema=GlobInput)
def glob(pattern: str, root: str) -> str:
    """Find files matching a glob pattern under the given root directory."""
    root_path = Path(root)
    if not root_path.exists():
        return f"Error: path_not_found: {root}"

    matches = sorted(root_path.glob(pattern))
    if not matches:
        return f"No files found matching '{pattern}' under {root}"

    # Return relative paths for readability
    lines = [str(p.relative_to(root_path)) for p in matches]
    return "\n".join(lines)


@tool(args_schema=GrepInput)
def grep(pattern: str, path: str, glob: str = "*") -> str:
    """Search for a regex pattern in files under the given path.

    Returns matching lines with file path and line number.
    """
    target = Path(path)
    if not target.exists():
        return f"Error: path_not_found: {path}"

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Error: invalid_regex: {e}"

    results = []

    files = [target] if target.is_file() else [
        f for f in target.rglob("*")
        if f.is_file() and fnmatch.fnmatch(f.name, glob)
    ]

    for file_path in sorted(files)[:100]:  # cap at 100 files
        try:
            for i, line in enumerate(
                file_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if regex.search(line):
                    rel = file_path.relative_to(target) if target.is_dir() else file_path
                    results.append(f"{rel}:{i}: {line.strip()}")
        except Exception:
            continue

    if not results:
        return f"No matches found for '{pattern}'"

    output = "\n".join(results[:200])  # cap output lines
    if len(results) > 200:
        output += f"\n\n[Output truncated. {len(results) - 200} more matches not shown.]"
    return output


@tool(args_schema=WriteFileInput)
def write_file(file_path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed.

    This is a high-risk operation — always requires human approval before execution.
    """
    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {path} ({len(content)} chars)"
    except Exception as e:
        return f"Error writing file: {e}"


@tool(args_schema=EditFileInput)
def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Perform an exact string replacement in a file.

    Use this for targeted updates — memory files, appending sections, small edits.
    old_string must appear exactly once in the file.
    """
    path = Path(file_path)
    if not path.exists():
        return f"Error: file_not_found: {file_path}"

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"

    count = content.count(old_string)
    if count == 0:
        return "Error: old_string not found in file"
    if count > 1:
        return f"Error: old_string found {count} times — must be unique. Add more surrounding context."

    path.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
    return f"Successfully edited {file_path}"


# Exported list for use in the agent
FILESYSTEM_TOOLS = [ls, read_file, glob, grep, write_file, edit_file]

# write_file (full overwrite) requires HITL; edit_file (targeted) does not
HIGH_RISK_TOOLS = {"write_file"}
