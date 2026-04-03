import ast
from pathlib import Path


class DataFrameInsertVisitor(ast.NodeVisitor):
    def __init__(self):
        self.issues = []

    def visit_Assign(self, node):  # noqa: N802
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "df"
            ):
                self.issues.append((node.lineno, "df[...] = detected"))
        self.generic_visit(node)


def scan_file(file_path: Path):
    try:
        tree = ast.parse(file_path.read_text())
    except Exception:
        return []

    visitor = DataFrameInsertVisitor()
    visitor.visit(tree)
    return visitor.issues


def main():
    root = Path(".")
    py_files = list(root.rglob("*.py"))

    total = 0

    for file in py_files:
        issues = scan_file(file)
        if issues:
            print(f"\n{file}")
            for lineno, msg in issues:
                print(f"  Line {lineno}: {msg}")
                total += 1

    print(f"\nTotal potential fragmentation spots: {total}")


if __name__ == "__main__":
    main()
