import os


def buscar_archivo_desesperadamente(nombre):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lugares = [
        os.path.join(base_dir, "data", "calibracion"),
        os.path.join(base_dir, "data"),
        base_dir,
        os.path.join(base_dir, ".."),
        os.path.join(base_dir, "..", "data", "calibracion"),
        os.getcwd()
    ]
    for ruta in lugares:
        path = os.path.join(ruta, nombre)
        if os.path.exists(path): return path
    for root, dirs, files in os.walk(base_dir):
        if nombre in files: return os.path.join(root, nombre)
    return None