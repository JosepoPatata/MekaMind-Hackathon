#!/usr/init/env python
"""
Clasificador dinámico de cubos con SO-101 controlado por REPLAY de LeRobot (subprocess).
Incluye reconocimiento de voz, alternancia automática Ida/Vuelta y modo Torre.
"""
import argparse
import json
import os
import subprocess
import time

import cv2
import numpy as np
import pandas as pd

import speech_recognition as sr

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

def clasificar_color_hsv(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        return "desconocido"

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_mean = np.median(hsv[:, :, 0])
    s_mean = np.median(hsv[:, :, 1])
    v_mean = np.median(hsv[:, :, 2])

    if s_mean < 50 or v_mean < 60:
        return "desconocido"

    if (0 <= h_mean <= 10) or (160 <= h_mean <= 179):
        return "rojo"
    elif 15 <= h_mean <= 35:
        return "amarillo"
    elif 35 < h_mean <= 85:
        return "verde"
    elif 90 <= h_mean <= 135:
        return "azul"

    return "desconocido"

# ==============================================================================
# 2. CARGA DE CALIBRACIÓN DE CÁMARA (ROIs)
# ==============================================================================
def cargar_rois_calibradas(archivo_json=r"C:\Users\LENOVO\AppData\Roaming\SenseCraft Robotics\backend-storage\calibration\lerobot\robots\so_follower"):
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
            
            print(f"[+] ROIs cargadas correctamente desde SenseCraft Robotics.")
            return rois
        except Exception as e:
            print(f"[!] Error al leer el archivo de calibración de SenseCraft: {e}.")

    print("[!] Usando ROIs por defecto predeterminadas.")
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
    def __init__(self, port, robot_id, repo_id_ida, repo_id_vuelta, repo_id_torre, episodios=None, simulacion=False):
        self.port = port
        self.robot_id = robot_id
        self.repo_id_ida = repo_id_ida
        self.repo_id_vuelta = repo_id_vuelta
        self.repo_id_torre = repo_id_torre
        self.episodios = episodios or EPISODIOS
        self.simulacion = simulacion

    def reproducir_episodio_directo(self, num_episodio, repo_activo, descripcion):
        if self.simulacion:
            print(f"\n[SIMULACIÓN - {descripcion}] Episodio {num_episodio} (Repo: {repo_activo})")
            time.sleep(1.0)
            return True

        print(f"\n[REPLAY - {descripcion}] Lanzando episodio {num_episodio} en {repo_activo}...")
        comando = [
            "lerobot-replay",
            "--robot.type=so101_follower",
            f"--robot.port={self.port}",
            f"--robot.id={self.robot_id}",
            f"--dataset.repo_id={repo_activo}",
            f"--dataset.episode={num_episodio}"
        ]

        try:
            subprocess.run(comando, text=True, check=True)
            print(f"[+] Replay de {descripcion} completado con éxito.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[!] Error en lerobot-replay:\n{e.stderr}")
        except FileNotFoundError:
            print("[!] Error: No se encontró el comando 'lerobot-replay'.")
        return False

    def reproducir(self, clave_origen_destino, es_vuelta=False):
        if clave_origen_destino not in self.episodios:
            print(f"[!] No hay un episodio asignado para {etiqueta(clave_origen_destino)}.")
            return False

        num_episodio = self.episodios[clave_origen_destino]
        repo_activo = self.repo_id_vuelta if es_vuelta else self.repo_id_ida
        descripcion = "VUELTA (Devolviendo bloque)" if es_vuelta else "IDA (Colocando bloque)"
        return self.reproducir_episodio_directo(num_episodio, repo_activo, descripcion)

# ==============================================================================
# 4. ESCANEO DINÁMICO DE LA MESA
# ==============================================================================
def escanear_mesa(frame, rois_mesa):
    mapa_posiciones_color = {}

    for pos_nombre, (x1, y1, x2, y2) in rois_mesa.items():
        roi = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
        cv2.putText(frame, pos_nombre, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        if roi.size == 0:
            continue

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = (hsv_roi[:, :, 1] > 45) & (hsv_roi[:, :, 2] > 55)
        
        if not np.any(mask):
            continue

        h_valid = hsv_roi[:, :, 0][mask]
        s_valid = hsv_roi[:, :, 1][mask]
        v_valid = hsv_roi[:, :, 2][mask]

        h_mean = np.median(h_valid)
        s_mean = np.median(s_valid)
        v_mean = np.median(v_valid)

        if s_mean < 45 or v_mean < 55:
            continue

        categoria = "desconocido"
        if (0 <= h_mean <= 10) or (160 <= h_mean <= 179):
            categoria = "rojo"
        elif 10 <= h_mean <= 42:
            categoria = "amarillo"
        elif 42 < h_mean <= 85:
            categoria = "verde"
        elif 90 <= h_mean <= 135:
            categoria = "azul"

        if categoria == "desconocido":
            continue

        mapa_posiciones_color[pos_nombre] = categoria
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, categoria.upper(), (x1, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return mapa_posiciones_color

def escuchar_voz(prompt):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"\n[MICRÓFONO ACTIVADO] {prompt}")
        r.adjust_for_ambient_noise(source, duration=0.5) 
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=3)
            texto = r.recognize_google(audio, language="es-ES")
            print(f"  [>] Has dicho: '{texto}'")
            return texto.strip().lower()
        except sr.WaitTimeoutError:
            print("  [!] No se detectó voz a tiempo.")
        except sr.UnknownValueError:
            print("  [!] No se ha entendido el audio.")
        except sr.RequestError as e:
            print(f"  [!] Error de red/API: {e}")
    return ""

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="COM8", help="Puerto del brazo follower")
    ap.add_argument("--robot-id", default="my_awesome_follower_arm", help="ID del robot")
    ap.add_argument("--repo-id", default="aitorvr06/bloques-transfer", help="Dataset principal de Ida")
    ap.add_argument("--repo-id-vuelta", default="aitorvr06/block-transfer-return", help="Dataset secundario de Vuelta")
    ap.add_argument("--repo-id-torre", default="aitorvr06/block-tower", help="Dataset para construir y deshacer la torre")
    ap.add_argument("--camara", type=int, default=1, help="Índice de la cámara")
    ap.add_argument("--simulacion", action="store_true", help="Modo simulación sin brazo físico")
    args = ap.parse_args()

    rois_mesa = cargar_rois_calibradas()
    brazo = BrazoReplaySubprocess(args.port, args.robot_id, args.repo_id, args.repo_id_vuelta, args.repo_id_torre, simulacion=args.simulacion)

    cap = cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        print(f"[!] Error: No se pudo abrir la cámara {args.camara}.")
        return

    print("\n==================================================")
    print(" SISTEMA MULTIMODO (IDA/VUELTA Y TORRE) ACTIVO")
    print(" Instrucciones:")
    print("   1. Pulsa [ESPACIO] y di un color o la palabra 'torre'.")
    print("   2. Vuelve a pulsar [ESPACIO] para la acción inversa.")
    print("   3. Pulsa [ESC] para salir.")
    print("==================================================\n")

    ultimo_movimiento = None 
    modo_actual = None  # Puede ser None, "esperando_vuelta_color", o "esperando_deshacer_torre"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            mapa_actual = escanear_mesa(frame, rois_mesa)

            # Mostrar estado dinámico en la pantalla de video
            if modo_actual is None:
                texto_estado = "ESTADO: Listo (Di color o 'torre') [ESPACIO]"
            elif modo_actual == "esperando_vuelta_color":
                texto_estado = "ESTADO: Listo para VUELTA DE COLOR [ESPACIO]"
            elif modo_actual == "esperando_deshacer_torre":
                texto_estado = "ESTADO: Listo para DESHACER TORRE [ESPACIO]"

            cv2.putText(frame, texto_estado, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            cv2.imshow("Monitor Dinamico - SO-101", frame)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                break

            if key == 32:  # Barra espaciadora
                if modo_actual is None:
                    print("\n--- NUEVA ORDEN (POR VOZ) ---")
                    comando = escuchar_voz("Habla ahora: ¿Qué color quieres mover o di 'torre'?")
                    if not comando:
                        print("[!] Operación abortada. Pulsa ESPACIO para reintentar.")
                        continue

                    # Comprobar si se ha dicho la palabra clave "torre"
                    if "torre" in comando:
                        print("[>] Orden detectada: CONSTRUIR TORRE")
                        exito = brazo.reproducir_episodio_directo(0, args.repo_id_torre, "TORRE (Construir)")
                        if exito:
                            modo_actual = "esperando_deshacer_torre"
                            print("[✓] Torre construida. Pulsa ESPACIO de nuevo para deshacerla.")
                            for _ in range(5):
                                cap.grab()
                    else:
                        # Flujo normal de colores (Ida)
                        color_origen = comando
                        color_destino = escuchar_voz("Habla ahora: ¿A qué color quieres llevarlo?")
                        if not color_destino:
                            print("[!] Operación abortada. Pulsa ESPACIO para reintentar.")
                            continue

                        colores_validos = ["rojo", "verde", "azul", "amarillo"]
                        color_origen = next((c for c in colores_validos if c in color_origen), color_origen)
                        color_destino = next((c for c in colores_validos if c in color_destino), color_destino)

                        pos_origen, pos_destino = None, None
                        for pos, col in mapa_actual.items():
                            if col == color_origen:
                                pos_origen = pos
                            elif col == color_destino:
                                pos_destino = pos

                        print(f"[>] Buscando bloque '{color_origen}' -> Posición origen: {pos_origen}")
                        print(f"[>] Buscando bloque destino '{color_destino}' -> Posición destino: {pos_destino}")

                        if pos_origen and pos_destino:
                            if pos_origen == pos_destino:
                                print("[!] Error: El bloque origen y destino están en la misma posición.")
                            else:
                                exito = brazo.reproducir((pos_origen, pos_destino), es_vuelta=False)
                                if exito:
                                    ultimo_movimiento = (pos_origen, pos_destino)
                                    modo_actual = "esperando_vuelta_color"
                                    print("[✓] Bloque colocado. Pulsa ESPACIO de nuevo cuando quieras devolverlo.")
                                    for _ in range(5):
                                        cap.grab()
                        else:
                            print("[!] No se han podido localizar ambos colores en la mesa en este momento.")

                elif modo_actual == "esperando_vuelta_color":
                    print("\n--- EJECUTANDO VUELTA AUTOMÁTICA DE COLOR ---")
                    if ultimo_movimiento:
                        exito = brazo.reproducir(ultimo_movimiento, es_vuelta=True)
                        if exito:
                            modo_actual = None
                            ultimo_movimiento = None
                            print("[✓] Bloque devuelto a su posición inicial con éxito.")
                            for _ in range(5):
                                cap.grab()
                    else:
                        print("[!] No hay registro del último movimiento.")

                elif modo_actual == "esperando_deshacer_torre":
                    print("\n--- EJECUTANDO DESHACER TORRE ---")
                    exito = brazo.reproducir_episodio_directo(1, args.repo_id_torre, "TORRE (Deshacer)")
                    if exito:
                        modo_actual = None
                        print("[✓] Torre deshecha con éxito.")
                        for _ in range(5):
                            cap.grab()
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()