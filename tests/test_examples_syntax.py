import os
import glob
import ast
import py_compile

import nbformat
import pytest

# Root directory of the repository (two levels up from this test file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
EXAMPLES_DIR = os.path.join(ROOT_DIR, "examples")

# -----------------------------------------------------------------------------
# Helper collectors
# -----------------------------------------------------------------------------

def _collect_example_py_files():
    """Return list of all *.py files under examples/ recursively."""
    pattern = os.path.join(EXAMPLES_DIR, "**", "*.py")
    return [path for path in glob.glob(pattern, recursive=True) if os.path.isfile(path)]


def _collect_example_notebooks():
    """Return list of all *.ipynb files under examples/ recursively."""
    pattern = os.path.join(EXAMPLES_DIR, "**", "*.ipynb")
    return [path for path in glob.glob(pattern, recursive=True) if os.path.isfile(path)]


# -----------------------------------------------------------------------------
# Tests for Python scripts
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("py_path", _collect_example_py_files())
def test_example_python_files_compile(py_path):
    """Ensure each example Python script has valid syntax.

    This uses ``py_compile`` so the file is parsed by CPython without execution
    of the module-level code, avoiding heavy runtime dependencies.
    """
    # doraise=True raises a ``py_compile.PyCompileError`` on failure which
    # pytest will treat as a test failure.
    py_compile.compile(py_path, doraise=True)


# -----------------------------------------------------------------------------
# Tests for Jupyter notebooks
# -----------------------------------------------------------------------------


def _clean_code(source: str) -> str:
    """Remove Jupyter magics / shell escapes so source can be parsed by ``ast``.

    Lines starting with ``%`` or ``!`` are stripped because they are not valid
    Python syntax outside a notebook environment.
    """
    cleaned_lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            # Skip IPython magic or shell command
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


@pytest.mark.parametrize("nb_path", _collect_example_notebooks())
def test_example_notebook_cells_parse(nb_path):
    """Validate that each code cell in the example notebooks can be parsed.

    Instead of executing potentially heavy code, we parse the cleaned source of
    each code cell with the ``ast`` module to ensure syntactic correctness.
    """
    nb = nbformat.read(nb_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        cleaned = _clean_code(cell.source)
        if cleaned.strip() == "":
            # Skip empty cells after cleaning
            continue
        # ``ast.parse`` raises ``SyntaxError`` on invalid Python code which will
        # fail the test if encountered.
        ast.parse(cleaned, filename=nb_path) 