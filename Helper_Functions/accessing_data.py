def h5tree_view(file):
    """Display HDF5 file structure in tree-like format.
    
    Parameters
    ----------
        :file: h5py.File , opened HDF5 file.
    """
    import h5py
    assert type(file) == h5py._hl.files.File

    def view_h5object(obj, depth=0):
        name = obj.name.split("/")[-1]
        deep = "│   "*depth + "├──"
        if type(obj) == h5py._hl.group.Group:
            print(deep,f"📁{name}")
            for k in list(obj):
                view_h5object(obj[k], depth=depth+1)
        else:
            print(deep,f"🔢{name}", f"⚙️{obj.shape}{obj.dtype}")

    print(".", file.filename)
    for k in list(file):
        view_h5object(file[k], depth=0)
