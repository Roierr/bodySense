"""Registro de sesiones: usuarios, coordenadas por cuadro y metricas posturales.

Tres piezas que se reparten el trabajo segun su tamano:

  - Las COORDENADAS crudas van a un CSV por sesion, una fila por instante.
    Son muchas (30 filas por segundo) y no se consultan, se abren en Excel.
  - Los USUARIOS y las SESIONES van a una base SQLite de un solo archivo.
    Son pocas y si se consultan ("las sesiones de Juan por fecha").
  - Las METRICAS posturales, que son el resumen de cada sesion, tambien van a
    la base, porque son lo que se compara entre personas.

Por que los angulos y no las coordenadas para comparar: las coordenadas de un
hombro dependen de la estatura de la persona y de donde se paro en la sala.
Dos personas con postura identica dan coordenadas distintas. Los angulos no:
no dependen ni del tamano ni de la posicion, y por eso son lo unico que se
puede comparar contra un usuario de referencia.

    python3 registro.py --check        prueba los angulos sin camara ni base
"""
import csv
import os
import sqlite3
import sys
from datetime import datetime

import numpy as np

from utils.Herramientas import BASE

ARCHIVO_BD = os.path.join(BASE, "bodysense.db")
CARPETA_SESIONES = os.path.join(BASE, "sesiones")

# Indices de MediaPipe Pose que necesitan las metricas.
NARIZ = 0
HOMBRO_IZQ, HOMBRO_DER = 11, 12
CADERA_IZQ, CADERA_DER = 23, 24

# Nombres legibles para las columnas del CSV. Los 33 puntos de MediaPipe, pero
# en el CSV se escriben todos para que el archivo sea autoexplicativo.
NOMBRES_PUNTOS = {
    0: "nariz", 1: "ojo_izq_int", 2: "ojo_izq", 3: "ojo_izq_ext",
    4: "ojo_der_int", 5: "ojo_der", 6: "ojo_der_ext", 7: "oreja_izq",
    8: "oreja_der", 9: "boca_izq", 10: "boca_der", 11: "hombro_izq",
    12: "hombro_der", 13: "codo_izq", 14: "codo_der", 15: "muneca_izq",
    16: "muneca_der", 17: "menique_izq", 18: "menique_der", 19: "indice_izq",
    20: "indice_der", 21: "pulgar_izq", 22: "pulgar_der", 23: "cadera_izq",
    24: "cadera_der", 25: "rodilla_izq", 26: "rodilla_der", 27: "tobillo_izq",
    28: "tobillo_der", 29: "talon_izq", 30: "talon_der", 31: "pie_izq",
    32: "pie_der",
}

# En el sistema que sale de la triangulacion, Y crece hacia ABAJO (convencion
# de imagenes), asi que "arriba" es -Y. Z es la distancia a las camaras.
ARRIBA = np.array([0.0, -1.0, 0.0])


# ==========================================
# METRICAS POSTURALES
# ==========================================
def _angulo(u, v):
    """Angulo en grados entre dos vectores. Devuelve None si alguno es nulo."""
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return None
    coseno = float(np.dot(u, v) / (nu * nv))
    return float(np.degrees(np.arccos(np.clip(coseno, -1.0, 1.0))))


def metricas_posturales(puntos):
    """Angulos posturales de un cuadro. puntos es {indice: [x, y, z]} en mm.

    Devuelve un dict con las metricas que se pudieron calcular; las que
    necesitan un punto que falta simplemente no aparecen.

    Las cuatro son invariantes a la estatura y a donde se paro la persona, que
    es lo que permite compararlas entre usuarios. Lo que NO son es clinicas:
    MediaPipe estima centros de articulacion, no marcas anatomicas, asi que
    los valores absolutos no equivalen a una medicion medica. Comparar dos
    sesiones del MISMO montaje si es valido; decir "esta persona tiene 12
    grados de escoliosis" no lo es.
    """
    m = {}
    p = {i: np.asarray(v, dtype=float) for i, v in puntos.items()}

    tiene = lambda *ids: all(i in p for i in ids)

    centro_hombro = centro_cadera = None
    if tiene(HOMBRO_IZQ, HOMBRO_DER):
        centro_hombro = (p[HOMBRO_IZQ] + p[HOMBRO_DER]) / 2
    if tiene(CADERA_IZQ, CADERA_DER):
        centro_cadera = (p[CADERA_IZQ] + p[CADERA_DER]) / 2

    # 1. Inclinacion del tronco: cuanto se aparta de la vertical el vector que
    #    va del centro de las caderas al centro de los hombros. 0 = derecho.
    #    Mide la inclinacion TOTAL: no distingue si es hacia adelante o de lado.
    if centro_hombro is not None and centro_cadera is not None:
        ang = _angulo(centro_hombro - centro_cadera, ARRIBA)
        if ang is not None:
            m["inclinacion_tronco"] = ang

    # 2. Desnivel de hombros: cuanto se aparta de la horizontal la linea entre
    #    los dos hombros. 0 = parejos. Es un angulo en 3D, asi que no cambia
    #    si la persona gira: solo mide si un hombro esta mas alto que el otro.
    if tiene(HOMBRO_IZQ, HOMBRO_DER):
        d = p[HOMBRO_IZQ] - p[HOMBRO_DER]
        largo = float(np.linalg.norm(d))
        if largo > 1e-6:
            m["desnivel_hombros"] = float(
                np.degrees(np.arcsin(np.clip(abs(d[1]) / largo, 0.0, 1.0))))

    # 3. Torsion del torso: cuanto estan girados los hombros respecto a las
    #    caderas, visto desde arriba. 0 = alineados.
    if tiene(HOMBRO_IZQ, HOMBRO_DER, CADERA_IZQ, CADERA_DER):
        plano = np.array([1.0, 0.0, 1.0])           # aplasta el eje vertical
        ang = _angulo((p[HOMBRO_IZQ] - p[HOMBRO_DER]) * plano,
                      (p[CADERA_IZQ] - p[CADERA_DER]) * plano)
        if ang is not None:
            m["torsion_torso"] = min(ang, 180.0 - ang)   # izq/der da lo mismo

    # 4. Adelantamiento de cabeza: que tan lejos esta la nariz de quedar
    #    justo encima de los hombros, en proporcion al alto del torso.
    #    Se divide entre el torso para que no dependa de la estatura.
    #    Mide distancia horizontal total: tampoco distingue adelante de lado.
    if NARIZ in p and centro_hombro is not None and centro_cadera is not None:
        alto_torso = float(np.linalg.norm(centro_hombro - centro_cadera))
        if alto_torso > 1e-6:
            d = p[NARIZ] - centro_hombro
            horizontal = float(np.hypot(d[0], d[2]))
            m["adelanto_cabeza"] = horizontal / alto_torso

    return m


def resumir(valores_por_metrica):
    """Mediana y dispersion de cada metrica a lo largo de una sesion.

    Mediana y no promedio, por lo mismo que en el resto del proyecto: un
    cuadro mal triangulado puede aterrizar lejisimos y arrastraria el promedio.
    p10 y p90 dicen que tan consistente estuvo la persona: un rango angosto es
    una postura sostenida, uno ancho es que se movio mucho.
    """
    resumen = {}
    for nombre, valores in valores_por_metrica.items():
        if not valores:
            continue
        a = np.asarray(valores, dtype=float)
        resumen[nombre] = {
            "mediana": float(np.median(a)),
            "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)),
            "muestras": int(a.size),
        }
    return resumen


# ==========================================
# BASE DE DATOS
# ==========================================
ESQUEMA = """
CREATE TABLE IF NOT EXISTS usuarios (
    id            INTEGER PRIMARY KEY,
    nombre        TEXT    NOT NULL UNIQUE,
    altura_cm     REAL,
    notas         TEXT,
    creado        TEXT    NOT NULL,
    es_referencia INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sesiones (
    id                 INTEGER PRIMARY KEY,
    usuario_id         INTEGER NOT NULL REFERENCES usuarios(id) ON DELETE CASCADE,
    inicio             TEXT    NOT NULL,
    fin                TEXT,
    cuadros            INTEGER NOT NULL DEFAULT 0,
    archivo_csv        TEXT,
    distancia_media_mm REAL
);

CREATE TABLE IF NOT EXISTS metricas (
    sesion_id INTEGER NOT NULL REFERENCES sesiones(id) ON DELETE CASCADE,
    nombre    TEXT    NOT NULL,
    mediana   REAL    NOT NULL,
    p10       REAL,
    p90       REAL,
    muestras  INTEGER NOT NULL,
    PRIMARY KEY (sesion_id, nombre)
);

CREATE INDEX IF NOT EXISTS idx_sesiones_usuario ON sesiones(usuario_id, inicio);
"""


def abrir_bd(ruta=None):
    """Abre (y crea si hace falta) la base. Devuelve la conexion."""
    con = sqlite3.connect(ruta or ARCHIVO_BD)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")   # sqlite las ignora si no se pide
    con.executescript(ESQUEMA)
    con.commit()
    return con


def crear_usuario(con, nombre, altura_cm=None, notas=None):
    """Alta de usuario. Devuelve su id; si ya existe, devuelve el que tenia.

    Todas las consultas usan parametros (?) y nunca texto armado a mano: el
    nombre lo escribe una persona y un nombre como  '); DROP TABLE usuarios;--
    borraria la base si se concatenara.
    """
    nombre = (nombre or "").strip()
    if not nombre:
        raise ValueError("el nombre del usuario no puede estar vacio")

    fila = con.execute("SELECT id FROM usuarios WHERE nombre = ?", (nombre,)).fetchone()
    if fila:
        return int(fila["id"])

    cur = con.execute(
        "INSERT INTO usuarios (nombre, altura_cm, notas, creado) VALUES (?, ?, ?, ?)",
        (nombre, altura_cm, notas, datetime.now().isoformat(timespec="seconds")))
    con.commit()
    return int(cur.lastrowid)


def marcar_referencia(con, usuario_id):
    """Deja a ese usuario como el unico de referencia."""
    con.execute("UPDATE usuarios SET es_referencia = 0")
    con.execute("UPDATE usuarios SET es_referencia = 1 WHERE id = ?", (usuario_id,))
    con.commit()


def usuario_de_referencia(con):
    return con.execute("SELECT * FROM usuarios WHERE es_referencia = 1").fetchone()


def metricas_de_sesion(con, sesion_id):
    filas = con.execute("SELECT * FROM metricas WHERE sesion_id = ?", (sesion_id,))
    return {f["nombre"]: dict(f) for f in filas}


def ultima_sesion(con, usuario_id):
    return con.execute(
        "SELECT * FROM sesiones WHERE usuario_id = ? AND cuadros > 0 "
        "ORDER BY inicio DESC LIMIT 1", (usuario_id,)).fetchone()


def comparar_con_referencia(con, sesion_id):
    """Diferencia de cada metrica contra la ultima sesion del usuario de referencia.

    Devuelve {metrica: {"valor", "referencia", "diferencia"}}. Vacio si no hay
    usuario de referencia o si no tiene sesiones.
    """
    ref = usuario_de_referencia(con)
    if ref is None:
        return {}
    ses_ref = ultima_sesion(con, ref["id"])
    if ses_ref is None or ses_ref["id"] == sesion_id:
        return {}

    mias = metricas_de_sesion(con, sesion_id)
    suyas = metricas_de_sesion(con, ses_ref["id"])
    salida = {}
    for nombre, m in mias.items():
        if nombre in suyas:
            salida[nombre] = {
                "valor": m["mediana"],
                "referencia": suyas[nombre]["mediana"],
                "diferencia": m["mediana"] - suyas[nombre]["mediana"],
            }
    return salida


# ==========================================
# SESION EN CURSO
# ==========================================
class Sesion:
    """Graba una sesion: CSV de coordenadas + fila en la base + metricas.

    Se usa asi:

        with Sesion(con, usuario_id) as s:
            s.anotar(t, puntos, distancia)      # una vez por cuadro
        print(s.resumen)

    El CSV se escribe cuadro por cuadro, no al final: si el programa se cae a
    la mitad, lo grabado hasta ahi queda en disco.
    """

    def __init__(self, con, usuario_id, carpeta=None):
        self.con = con
        self.usuario_id = usuario_id
        self.inicio = datetime.now()
        self.cuadros = 0
        self.distancias = []
        self._valores = {}
        self.resumen = {}

        carpeta = carpeta or CARPETA_SESIONES
        os.makedirs(carpeta, exist_ok=True)
        marca = self.inicio.strftime("%Y%m%d_%H%M%S")
        self.ruta_csv = os.path.join(carpeta, f"usuario{usuario_id}_{marca}.csv")

        cur = con.execute(
            "INSERT INTO sesiones (usuario_id, inicio, archivo_csv) VALUES (?, ?, ?)",
            (usuario_id, self.inicio.isoformat(timespec="seconds"), self.ruta_csv))
        con.commit()
        self.id = int(cur.lastrowid)

        self._f = open(self.ruta_csv, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._f)
        cabecera = ["cuadro", "t_seg", "distancia_mm"]
        for i in range(33):
            nombre = NOMBRES_PUNTOS.get(i, f"punto{i}")
            cabecera += [f"{nombre}_x", f"{nombre}_y", f"{nombre}_z"]
        self._csv.writerow(cabecera)

    def anotar(self, t_seg, puntos, distancia_mm=None):
        """Guarda un cuadro. puntos es {indice: [x, y, z]} en mm; puede faltar."""
        fila = [self.cuadros, round(float(t_seg), 3),
                "" if distancia_mm is None else round(float(distancia_mm), 1)]
        for i in range(33):
            p = puntos.get(i)
            # Se redondea a un decimal de milimetro: la precision real del
            # sistema es de ~9 mm, asi que mas decimales son precision falsa
            # que solo infla el archivo.
            fila += ["", "", ""] if p is None else [round(float(v), 1) for v in p]
        self._csv.writerow(fila)

        if distancia_mm is not None:
            self.distancias.append(float(distancia_mm))
        for nombre, valor in metricas_posturales(puntos).items():
            self._valores.setdefault(nombre, []).append(valor)
        self.cuadros += 1

    def cerrar(self):
        if self._f.closed:
            return
        self._f.close()
        self.resumen = resumir(self._valores)

        self.con.execute(
            "UPDATE sesiones SET fin = ?, cuadros = ?, distancia_media_mm = ? WHERE id = ?",
            (datetime.now().isoformat(timespec="seconds"), self.cuadros,
             float(np.median(self.distancias)) if self.distancias else None,
             self.id))
        self.con.executemany(
            "INSERT OR REPLACE INTO metricas "
            "(sesion_id, nombre, mediana, p10, p90, muestras) VALUES (?, ?, ?, ?, ?, ?)",
            [(self.id, n, r["mediana"], r["p10"], r["p90"], r["muestras"])
             for n, r in self.resumen.items()])
        self.con.commit()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cerrar()
        return False


# ==========================================
# CHEQUEO SIN CAMARA NI BASE
# ==========================================
def cuerpo_sintetico(inclinacion=0.0, desnivel=0.0, torsion=0.0, altura_mm=1700.0):
    """Un cuerpo de pie con los defectos posturales que se le pidan, en grados.

    Sirve para verificar las metricas contra una respuesta conocida: si le
    pido 15 grados de inclinacion, metricas_posturales tiene que devolver 15.
    """
    e = altura_mm / 1700.0
    ancho_hombro, ancho_cadera = 200.0 * e, 120.0 * e
    alto_torso = 500.0 * e

    # El desnivel se aplica ROTANDO la direccion de los hombros, no subiendo un
    # hombro: subirlo dejando fijo el ancho alarga la linea y el angulo real
    # sale menor que el pedido (5 grados pedidos daban 4.98 medidos).
    rt, rd = np.radians(torsion), np.radians(desnivel)
    hombro_dir = np.array([np.cos(rt) * np.cos(rd), -np.sin(rd),
                           np.sin(rt) * np.cos(rd)])
    cadera_dir = np.array([1.0, 0.0, 0.0])

    centro_cadera = np.array([0.0, 0.0, 2000.0])
    # Y crece hacia abajo, asi que subir el torso es restar en Y.
    ri = np.radians(inclinacion)
    centro_hombro = centro_cadera + np.array(
        [alto_torso * np.sin(ri), -alto_torso * np.cos(ri), 0.0])

    return {
        NARIZ: centro_hombro + np.array([0.0, -180.0 * e, 0.0]),
        HOMBRO_IZQ: centro_hombro + hombro_dir * (ancho_hombro / 2),
        HOMBRO_DER: centro_hombro - hombro_dir * (ancho_hombro / 2),
        CADERA_IZQ: centro_cadera + cadera_dir * (ancho_cadera / 2),
        CADERA_DER: centro_cadera - cadera_dir * (ancho_cadera / 2),
    }


def demo():
    """Verifica las metricas contra cuerpos con defectos conocidos."""
    recto = metricas_posturales(cuerpo_sintetico())
    assert abs(recto["inclinacion_tronco"]) < 0.01, recto
    assert abs(recto["desnivel_hombros"]) < 0.01, recto
    assert abs(recto["torsion_torso"]) < 0.01, recto
    assert recto["adelanto_cabeza"] < 0.01, recto

    for pedido in (5.0, 15.0, 30.0):
        m = metricas_posturales(cuerpo_sintetico(inclinacion=pedido))
        assert abs(m["inclinacion_tronco"] - pedido) < 0.01, \
            f"inclinacion: pedi {pedido}, dio {m['inclinacion_tronco']:.2f}"
        m = metricas_posturales(cuerpo_sintetico(desnivel=pedido))
        assert abs(m["desnivel_hombros"] - pedido) < 0.01, \
            f"desnivel: pedi {pedido}, dio {m['desnivel_hombros']:.2f}"
        m = metricas_posturales(cuerpo_sintetico(torsion=pedido))
        assert abs(m["torsion_torso"] - pedido) < 0.01, \
            f"torsion: pedi {pedido}, dio {m['torsion_torso']:.2f}"

    # Invariancia a la estatura: es LA razon de usar angulos y no coordenadas.
    chico = metricas_posturales(cuerpo_sintetico(inclinacion=12.0, altura_mm=1400.0))
    grande = metricas_posturales(cuerpo_sintetico(inclinacion=12.0, altura_mm=1950.0))
    assert abs(chico["inclinacion_tronco"] - grande["inclinacion_tronco"]) < 0.01, \
        "la inclinacion no deberia depender de la estatura"
    assert abs(chico["adelanto_cabeza"] - grande["adelanto_cabeza"]) < 1e-6, \
        "el adelanto de cabeza esta normalizado: no deberia depender de la estatura"

    # Un cuadro incompleto no revienta: devuelve solo lo que pudo calcular.
    parcial = metricas_posturales({HOMBRO_IZQ: [100, 0, 2000], HOMBRO_DER: [-100, 0, 2000]})
    assert "desnivel_hombros" in parcial and "inclinacion_tronco" not in parcial, parcial

    print("OK: 4 metricas verificadas contra cuerpos sinteticos, "
          "invariantes a la estatura, y toleran cuadros incompletos")


if __name__ == "__main__":
    if "--check" in sys.argv:
        demo()
    else:
        print(__doc__)
        print(f"Base de datos: {ARCHIVO_BD}")
        print(f"Sesiones:      {CARPETA_SESIONES}")
