"""Paquete de visualizacion: actualmente vacío.

Se mantiene el paquete para compatibilidad con imports, pero su
implementacion de render 3D fue retirada para simplificar el proyecto.
"""

# Export minimal stubs para evitar ImportError en otros modulos.
def dibujar_hueso(*args, **kwargs):
	pass

def dibujar_joint(*args, **kwargs):
	pass

def dibujar_cabeza(*args, **kwargs):
	pass

__all__ = ["Graficos", "probar_render"]
