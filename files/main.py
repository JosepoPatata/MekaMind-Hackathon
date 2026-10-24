#!/usr/bin/env python
"""
Clasificador dinámico de cubos con SO-101 controlado por REPLAY de LeRobot (subprocess).

Lógica dinámica:
1. Lee las ROIs calibradas desde 'calibracion_completa.json'.
2. Pide por consola el color del bloque a mover y el color del bloque destino.
3. Escanea la mesa para localizar en qué posiciones están esos colores.
4. Ejecuta 'lerobot-replay' con el episodio correspondiente (0 al 11) dentro del dataset unificado.

Uso:
    python main.py --port COM3 --repo-id TU_USUARIO/so101_clasificador
    python main.py --simulacion     # Modo pruebas sin brazo físico
"""
import argparse
import json
import os
import subprocess
import time

import cv2
import numpy as np
import pandas as pd

# ==============================================================================
# 0. MAPEO DE LOS 12 EPISODIOS DENTRO DEL DATASET UNIFICADO
# ==============================================================================
EPISODIOS = {
    ("posicion_1", "posicion_2"): 0,
    ("posicion_1", "posicion_3"): 1,
    ("posicion_1", "posicion_4"): 2,
    ("posicion_2", "posicion_1"): 3,
    ("posicion_2", "posicion_3"): 4,
    ("posicion_2", "posicion_4"): 5,
    ("posicion_3", "posicion_1"): 6,
    ("posicion_3", "posicion_2"): 7,
    ("posicion_3", "posicion_4"): 8,
    ("posicion_4", "posicion_1"): 9,
    ("posicion_4", "posicion_2"): 10,
    ("posicion_4", "posicion_3"): 11,
}

def etiqueta(clave):
    return f"{clave[0]} -> {clave[1]}"

# ==============================================================================
# 1. BASE DE DATOS DE COLORES (colors.csv)
# ==============================================================================
import os

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "colors.csv")
    print(f"[*] Buscando 'colors.csv' en: {csv_path}")
    
    index = ["color", "color_name", "hex", "R", "G", "B"]
    csv = pd.read_csv(csv_path, names=index, header=None)
    colors_matrix = csv[["R", "G", "B"]].to_numpy()
    color_names = csv["color_name"].to_numpy()
    print("[+] ¡colors.csv cargado con éxito!")
except Exception as e:
    print(f"[!] Advertencia: No se pudo cargar 'colors.csv' ({e}).")
    colors_matrix = None

def get_color_name_fast(r, g, b):
    if colors_matrix is None:
        return "desconocido"
    distances = np.sum(np.abs(colors_matrix - [r, g, b]), axis=1)
    return color_names[np.argmin(distances)]

def clasificar_color(nombre_color):
    nombre = nombre_color.lower()
    
    if any(k in nombre for k in ["red", "crimson", "scarlet", "rose", "rojo", "maroon"]):
        return "rojo"
    if any(k in nombre for k in ["blue", "cyan", "navy", "azure", "azul"]):
        return "azul"
    if any(k in nombre for k in ["green", "lime", "olive", "emerald", "verde"]):
        return "verde"
    if any(k in nombre for k in ["yellow", "gold", "amber", "amarillo"]):
        return "amarillo"
        
    return "desconocido"

# ==============================================================================
# 2. CARGA DE CALIBRACIÓN DE CÁMARA (ROIs)
# ==============================================================================
def cargar_rois_calibradas(archivo_json="posiciones_config.json"):
    rois = {}
    if os.path.exists(archivo_json):
        try:
            with open(archivo_json, "r") as f:
                datos = json.load(f)
            for i in range(1, 5):
                clave_json = f"pos_{i}"
                if clave_json in datos and datos[clave_json]:
                    c = datos[clave_json]
                    rois[f"posicion_{i}"] = (c["xmin"], c["ymin"], c["xmax"], c["ymax"])
            print(f"[+] ROIs de posiciones cargadas desde '{archivo_json}'.")
            return rois
        except Exception as e:
            print(f"[!] Error al leer '{archivo_json}': {e}.")

    return {
        "posicion_1": (467, 222, 491, 248),
        "posicion_2": (392, 221, 416, 246),
        "posicion_3": (309, 229, 334, 250),
        "posicion_4": (220, 224, 242, 248),
    }

FRACCION_AREA_MIN = 0.25

# ==============================================================================
# 3. CONTROL DE LEROBOT-REPLAY VÍA SUBPROCESS
# ==============================================================================
class BrazoReplaySubprocess:
    def __init__(self, port, robot_id, repo_id, episodios=None, simulacion=False):
        self.port = port
        self.robot_id = robot_id
        self.repo_id = repo_id
        self.episodios = episodios or EPISODIOS
        self.simulacion = simulacion

    def reproducir(self, clave_origen_destino):
        if clave_origen_destino not in self.episodios:
            print(f"[!] No hay un episodio asignado para {etiqueta(clave_origen_destino)}.")
            return

        num_episodio = self.episodios[clave_origen_destino]

        if self.simulacion:
            print(f"[SIMULACIÓN] Replay ejecutado: {etiqueta(clave_origen_destino)} -> Episodio {num_episodio}")
            time.sleep(1.0)
            return

        print(f"\n[REPLAY] Lanzando 'lerobot-replay' para {etiqueta(clave_origen_destino)} (Episodio {num_episodio})...")
        comando = [
            "lerobot-replay",
            "--robot.type=so101_follower",
            f"--robot.port={self.port}",
            f"--robot.id={self.robot_id}",
            f"--dataset.repo_id={self.repo_id}",
            f"--dataset.episode={num_episodio}"
        ]

        try:
            subprocess.run(comando, text=True, check=True)
            print(f"[+] Replay completado con éxito.")
        except subprocess.CalledProcessError as e:
            print(f"[!] Error en lerobot-replay:\n{e.stderr}")
        except FileNotFoundError:
            print("[!] Error: No se encontró el comando 'lerobot-replay'.")

# ==============================================================================
# 4. ESCANEO DINÁMICO DE LA MESA
# ==============================================================================
def escanear_mesa(frame, rois_mesa):
    """Analiza las 4 posiciones y devuelve un diccionario {posicion: color}, ignorando grises/fondos."""
    mapa_posiciones_color = {}

    for pos_nombre, (x1, y1, x2, y2) in rois_mesa.items():
        roi = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
        cv2.putText(frame, pos_nombre, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        if roi.size == 0:
            continue

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_roi, 65, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_min = FRACCION_AREA_MIN * roi.shape[0] * roi.shape[1]

        for cnt in contours:
            if cv2.contourArea(cnt) <= area_min:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            parche = roi[max(0, cy - 5):cy + 5, max(0, cx - 5):cx + 5]
            if parche.size == 0:
                continue
            b, g, r = (int(v) for v in parche.reshape(-1, 3).mean(axis=0))

            categoria = clasificar_color(get_color_name_fast(r, g, b))

            # SI ES DESCONOCIDO (GRISES, FONDO DE MESA, ETC.), LO IGNORAMOS
            if categoria == "desconocido":
                continue

            mapa_posiciones_color[pos_nombre] = categoria

            # Dibujar el rectángulo y el nombre solo si es un color válido
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, categoria.upper(), (x1, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return mapa_posiciones_color

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM8", help="Puerto del brazo follower")
    ap.add_argument("--robot-id", default="my_awesome_follower_arm", help="ID del robot")
    ap.add_argument("--repo-id", default="aitorvr06/bloques-transfer", help="Dataset unificado en Hugging Face")
    ap.add_argument("--camara", type=int, default=1, help="Índice de la cámara")
    ap.add_argument("--simulacion", action="store_true", help="Modo simulación sin brazo físico")
    args = ap.parse_args()

    rois_mesa = cargar_rois_calibradas()
    brazo = BrazoReplaySubprocess(args.port, args.robot_id, args.repo_id, simulacion=args.simulacion)

    cap = cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        print(f"[!] Error: No se pudo abrir la cámara {args.camara}.")
        return

    print("\n==================================================")
    print(" SISTEMA DINÁMICO DE PICK & PLACE ACTIVO")
    print(" Instrucciones:")
    print("   1. Mantén la ventana de la cámara abierta.")
    print("   2. Escribe en la consola los colores cuando se te pida.")
    print("   3. Pulsa [ESC] en la ventana de video para salir.")
    print("==================================================\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Escanear la mesa en tiempo real para mostrar el estado visual
            mapa_actual = escanear_mesa(frame, rois_mesa)

            cv2.imshow("Monitor Dinamico - SO-101", frame)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # Tecla ESC para salir
                break

            # Si pulsas la barra espaciadora, el script te pregunta el comando por consola
            if key == 32:
                print("\n--- NUEVA ORDEN DE MOVIMIENTO ---")
                color_origen = input("¿Qué color de bloque quieres mover? (ej: rojo, azul, verde, amarillo): ").strip().lower()
                color_destino = input("¿Sobre qué color de bloque quieres colocarlo?: ").strip().lower()

                # Buscar en qué posición está cada color según el último escaneo de la mesa
                pos_origen = None
                pos_destino = None

                for pos, col in mapa_actual.items():
                    if col == color_origen:
                        pos_origen = pos
                    elif col == color_destino:
                        pos_destino = pos

                print(f"Buscando '{color_origen}' -> Encontrado en: {pos_origen}")
                print(f"Buscando '{color_destino}' -> Encontrado en: {pos_destino}")

                if pos_origen and pos_destino:
                    if pos_origen == pos_destino:
                        print("[!] Error: El bloque origen y destino están en la misma posición.")
                    else:
                        brazo.reproducir((pos_origen, pos_destino))
                        for _ in range(5):
                            cap.grab()
                else:
                    print("[!] No se han podido localizar ambos colores en la mesa. Comprueba que los bloques estén bien colocados.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()