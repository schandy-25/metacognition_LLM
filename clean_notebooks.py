import nbformat as nbf

paths = ["ethics.ipynb", "meta3.ipynb"]

for path in paths:
    print(f"Cleaning {path} ...")
    nb = nbf.read(path, as_version=nbf.NO_CONVERT)

    # Remove widgets metadata if present
    if "widgets" in nb.metadata:
        nb.metadata.pop("widgets", None)

    for cell in nb.cells:
        if "widgets" in cell.metadata:
            cell.metadata.pop("widgets", None)

    nbf.write(nb, path)
    print(f"Saved cleaned notebook: {path}")
