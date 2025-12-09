import tkinter as tk
from tkinter import colorchooser, messagebox
import json
import os

# Configuración segura por defecto
datos_avatar = {
    "nombre": "Jugador 1",
    "color_piel": "#3498db", 
    "grosor": 3, 
    "tamano_cabeza": 1.0
}

# Cargar configuración existente si está disponible
class InterfazCreador:
    def __init__(self, root):
        self.root = root
        self.root.title("CAMBIA EL COLOR DE TU AVATAR")
        self.root.geometry("350x300")
        self.color_actual = datos_avatar["color_piel"]

        # Título

        tk.Label(root, text="CONFIGURAR AVATAR", font=("Arial", 14, "bold")).pack(pady=20)
        
        # Nombre
        tk.Label(root, text="Nombre:").pack()
        self.var_nombre = tk.StringVar(value=datos_avatar["nombre"])
        tk.Entry(root, textvariable=self.var_nombre).pack(pady=5)

        # Color
        self.btn_color = tk.Button(root, text="🎨 Cambiar Color", bg=self.color_actual, fg="white", font=("Arial", 11), command=self.cambiar_color)
        self.btn_color.pack(pady=20, fill=tk.X, padx=40)

        # Guardar
        tk.Button(root, text="💾 GUARDAR Y USAR", bg="green", fg="white", font=("Arial", 12, "bold"), command=self.guardar).pack(pady=10, fill=tk.X, padx=20)


# Cambiar color de piel
    def cambiar_color(self):
        color = colorchooser.askcolor(color=self.color_actual)[1]
        if color:
            self.color_actual = color
            self.btn_color.config(bg=color)


# Guardar configuración
    def guardar(self):
        datos_avatar["nombre"] = self.var_nombre.get()
        datos_avatar["color_piel"] = self.color_actual
        
        # Guardar en archivo JSON
        base = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base, "avatar_config.json"), "w") as f:
            json.dump(datos_avatar, f)
        
        # Notificar al usuario
        messagebox.showinfo("Listo", "Guardado. ¡Abre el main.py!")
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = InterfazCreador(root)
    root.mainloop()