import onnxruntime as ort
import numpy as np
import cv2
import time
import win32api, win32con
import pandas as pd
import torch
import torchvision
import ctypes
import pygetwindow
import bettercam
import os
import json

GREEN = "\033[32m"  # Cor verde
RED = "\033[31m"    # Cor vermelha
RESET = "\033[0m"   # Resetando a cor para a padrão

# Configurações básicas
screenShotHeight = 320
screenShotWidth = 320

DEFAULT_CONFIG_FILE = "default_config.json"
USER_CONFIG_FILE = "user_config.json"

# Função para carregar as configurações padrão
def load_default_config():
    default_config = {
        "selected_model": "fortnite",
        "aaMovementAmp": 0.4,
        "headshot_mode": True,
        "target_fps": 60,
        "onnxChoice": 2,
        "confidence": 0.4,
        "useMask": False,
        "maskSide": "left",
        "maskWidth": 80,
        "maskHeight": 200,
        "cpsDisplay": True,
        "visuals": False,
        "centerOfScreen": True,
        "aaQuitKey": "p"
    }
    if not os.path.exists(DEFAULT_CONFIG_FILE):
        with open(DEFAULT_CONFIG_FILE, "w") as f:
            json.dump(default_config, f, indent=4)
    return default_config

# Função para carregar as configurações do usuário
def load_user_config():
    if not os.path.exists(USER_CONFIG_FILE):
        config = load_default_config()
        save_user_config(config)
    else:
        with open(USER_CONFIG_FILE, "r") as f:
            config = json.load(f)
    return config

# Função para salvar as configurações do usuário
def save_user_config(config):
    with open(USER_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

# Função para redefinir as configurações para os valores padrão
def reset_to_default():
    default_config = load_default_config()
    save_user_config(default_config)
    print("Configurações resetadas para o padrão.")

# Função para editar uma variável no arquivo de configuração
def edit_config(config_data, descriptions):
    print("\nConfigurações Atuais:")
    for key, value in config_data.items():
        if key in descriptions:
            print(f"{key}: {value}")
            print(f"Descrição: {descriptions.get(key, 'Sem descrição disponível.')}\n")
    
    key = input("\nDigite o nome da chave que deseja editar (ou 'sair' para sair): ")
    
    if key == 'sair':
        return False
    
    if key not in descriptions:
        print("Chave inválida! Você só pode editar as chaves listadas.")
        return True

    new_value = input(f"Digite o novo valor para '{key}' ({config_data[key]}): ")

    # Se for um valor booleano, converte para bool
    if isinstance(config_data[key], bool):
        new_value = new_value.lower() in ['true', '1', 'sim', 'verdadeiro']
    # Se for um valor inteiro, converte para int
    elif isinstance(config_data[key], int):
        new_value = int(new_value)
    # Se for um valor float, converte para float
    elif isinstance(config_data[key], float):
        new_value = float(new_value)
    # Caso contrário, assume que é uma string
    config_data[key] = new_value

    save_user_config(config_data)
    print(f"Configuração '{key}' atualizada para {new_value}.")
    return True

def main():
    # Carregar as configurações
    config_data = load_user_config()

    # Descrições de variáveis que o usuário pode editar
    descriptions = {
        "aaMovementAmp": "Multiplicador de movimento para aimbot (0.0 a 1.0).",
        "headshot_mode": "Ativar ou desativar o modo de cabeça (True/False).",
        "target_fps": "Frames por segundo alvo para o aimbot.",
        "onnxChoice": "Escolha de modelo ONNX, afeta a performance (1=CPU 1, 2=GPU AMD 3=GPU NVIDIA).",
        "confidence": "Valor de confiança mínimo para disparar no alvo (0.0 a 1.0).",
        "cpsDisplay": "Exibir o CPS (Captures por Segundo) no HUD (True/False).",
        "aaQuitKey": "Tecla para sair do modo de aimbot (ex: 'p')."
    }

    while True:
        print("\nEscolha uma opção:")
        print("1. Editar configurações")
        print("2. Resetar configurações para o padrão")
        print("3. Sair")

        choice = input("Digite o número da opção: ")

        if choice == '1':
            while edit_config(config_data, descriptions):
                pass
        elif choice == '2':
            reset_to_default()
            config_data = load_user_config()  # Recarrega após reset
        elif choice == '3':
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def display_title():
    clear()
    # Exibe o título em verde (ANSI escape code)
    print("\033[92m" + """
    
____    ____  _______ .__   __.  __    __       _______.        ___       __  
\   \  /   / |   ____||  \ |  | |  |  |  |     /       |       /   \     |  | 
 \   \/   /  |  |__   |   \|  | |  |  |  |    |   (----`      /  ^  \    |  | 
  \      /   |   __|  |  . `  | |  |  |  |     \   \         /  /_\  \   |  | 
   \    /    |  |____ |  |\   | |  `--'  | .----)   |       /  _____  \  |  | 
    \__/     |_______||__| \__|  \______/  |_______/       /__/     \__\ |__| 
                                                                              

""" + "\033[0m")

models = {
    "Fortnite": "Fortnite Model",
}

with open('user_config.json', 'r') as file:
    config = json.load(file)

# Agora você pode acessar as chaves dinamicamente
selected_model = config["selected_model"]
aaMovementAmp = config["aaMovementAmp"]
headshot_mode = config["headshot_mode"]
targett_fps = config["target_fps"]
onnxChoice = config["onnxChoice"]
confidence = config["confidence"]
useMask = config["useMask"]
maskSide = config["maskSide"]
maskWidth = config["maskWidth"]
maskHeight = config["maskHeight"]
cpsDisplay = config["cpsDisplay"]
visuals = config["visuals"]
centerOfScreen = config["centerOfScreen"]
aaQuitKey = config["aaQuitKey"]

# Funções auxiliares
def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,
):
    # Simplificado: só usa print no aviso de tempo excedido
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    device = prediction.device
    bs = prediction.shape[0]
    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres

    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 + 0.05 * bs
    merge = False

    t = time.time()
    mi = 5 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit:.3f}s exceeded")
            break
    return output

def gameSelection(target_fps):
    try:
        videoGameWindows = pygetwindow.getAllWindows()
        print("=== Todas as Janelas ===")
        for index, window in enumerate(videoGameWindows):
            if window.title != "":
                print(f"[{index}]: {window.title}")
        try:
            userInput = int(input("Escolha o número da janela: "))
        except ValueError:
            print("Número inválido!")
            return None
        videoGameWindow = videoGameWindows[userInput]
    except Exception as e:
        print("Erro ao selecionar a janela:", e)
        return None

    activationRetries = 30
    activationSuccess = False
    while activationRetries > 0:
        try:
            videoGameWindow.activate()
            activationSuccess = True
            break
        except Exception as e:
            print("Falha ao ativar a janela:", e)
        time.sleep(3.0)
        activationRetries -= 1
    if not activationSuccess:
        return None

    print("Janela ativada com sucesso!")
    left = ((videoGameWindow.left + videoGameWindow.right) // 2) - (screenShotWidth // 2)
    top = videoGameWindow.top + (videoGameWindow.height - screenShotHeight) // 2
    region = (left, top, left + screenShotWidth, top + screenShotHeight)
    print("Região:", region)
    camera = bettercam.create(region=region, output_color="BGRA", max_buffer_len=512)
    if camera is None:
        print("Erro na câmera!")
        return None
    camera.start(target_fps=target_fps, video_mode=True)
    return camera, screenShotWidth // 2, screenShotHeight // 2

# Estrutura para envio de movimento do mouse
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

class _INPUTunion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]

class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", ctypes.c_ulong),
                ("u", _INPUTunion)]

def send_mouse_move(rel_x, rel_y):
    inp = INPUT()
    inp.type = 0
    inp.mi.dx = int(rel_x)
    inp.mi.dy = int(rel_y)
    inp.mi.dwFlags = 0x0001
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))

def main(targett_fps):
    sel = gameSelection(target_fps=targett_fps)
    if sel is None:
        return
    camera, cWidth, cHeight = sel
    count = 0
    sTime = time.time()
    last_mid_coord = None
    aimbot_active = False

    # Configuração do provider ONNX
    if onnxChoice == 1:
        onnxProvider = "CPUExecutionProvider"
    elif onnxChoice == 2:
        onnxProvider = "DmlExecutionProvider"
    elif onnxChoice == 3:
        onnxProvider = "CUDAExecutionProvider"

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_sess = ort.InferenceSession('models/best.onnx', sess_options=so, providers=[onnxProvider])
    COLORS = np.random.uniform(0, 255, size=(1500, 3))

    while win32api.GetAsyncKeyState(ord(aaQuitKey)) == 0:
        if win32api.GetAsyncKeyState(win32con.VK_F2) & 0x0001:
            aimbot_active = not aimbot_active
            print(f"{GREEN}Aimbot ligado{RESET}" if aimbot_active else f"{RED}Aimbot desligado{RESET}", end="\r")
            time.sleep(0.2)

        # Pega o frame atual
        npImg = np.array(camera.get_latest_frame())
        if useMask:
            if maskSide.lower() == "right":
                npImg[-maskHeight:, -maskWidth:, :] = 0
            elif maskSide.lower() == "left":
                npImg[-maskHeight:, :maskWidth, :] = 0

        # Só roda a detecção se o aimbot estiver ativo
        if aimbot_active:
            if onnxChoice == 3:
                im = torch.from_numpy(npImg).to('cuda')
                if im.shape[2] == 4:
                    im = im[:, :, :3]
                im = torch.movedim(im, 2, 0)
                im = im.half() / 255
                if len(im.shape) == 3:
                    im = im.unsqueeze(0)
            else:
                im = np.array([npImg])
                if im.shape[3] == 4:
                    im = im[:, :, :, :3]
                im = im / 255
                im = im.astype(np.half)
                im = np.moveaxis(im, 3, 1)
            if onnxChoice == 3:
                # Se necessário, adapte para Nvidia
                outputs = ort_sess.run(None, {'images': im.cpu().numpy()})
            else:
                outputs = ort_sess.run(None, {'images': im})
            im_out = torch.from_numpy(outputs[0]).to('cpu')
            pred = non_max_suppression(im_out, confidence, confidence, max_det=10)
            targets = []
            for det in pred:
                if len(det):
                    for *xyxy, conf, cls in reversed(det):
                        norm_box = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / torch.tensor([screenShotWidth, screenShotHeight, screenShotWidth, screenShotHeight])).view(-1)
                        targets.append(norm_box.tolist() + [float(conf)])
            targets = pd.DataFrame(targets, columns=['current_mid_x', 'current_mid_y', 'width', "height", "confidence"]) if targets else pd.DataFrame()
            if not targets.empty:
                if centerOfScreen:
                    targets["dist_from_center"] = np.sqrt((targets.current_mid_x * screenShotWidth - cWidth) ** 2 +
                                                        (targets.current_mid_y * screenShotHeight - cHeight) ** 2)
                    targets = targets.sort_values("dist_from_center")

    # Convertendo coordenadas normalizadas para pixels
                xMid = targets.iloc[0].current_mid_x * screenShotWidth
                yMid = targets.iloc[0].current_mid_y * screenShotHeight
                box_height = targets.iloc[0].height * screenShotHeight

                headshot_offset = box_height * (0.38 if headshot_mode else 0.2)

                # Corrigindo cálculo de movimento do mouse
                mouseMove = [xMid - cWidth, (yMid - headshot_offset) - cHeight]

                if win32api.GetAsyncKeyState(win32con.VK_RBUTTON) & 0x8000:
                    move_x = int(mouseMove[0] * aaMovementAmp)
                    move_y = int(mouseMove[1] * aaMovementAmp)
                    send_mouse_move(move_x, move_y)

                last_mid_coord = [xMid, yMid]
            else:
                last_mid_coord = None


        if visuals:
            for i in range(len(targets) if not targets.empty else 0):
                halfW = round(targets["width"].iloc[i] / 2)
                halfH = round(targets["height"].iloc[i] / 2)
                midX = targets['current_mid_x'].iloc[i]
                midY = targets['current_mid_y'].iloc[i]
                startX, startY = int(midX - halfW), int(midY - halfH)
                endX, endY = int(midX + halfW), int(midY + halfH)
                label = f"Human: {targets['confidence'].iloc[i]*100:.2f}%"
                cv2.rectangle(npImg, (startX, startY), (endX, endY), COLORS[i % len(COLORS)], 2)
                y_txt = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(npImg, label, (startX, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[i % len(COLORS)], 2)
            cv2.imshow('Live Feed', npImg)
            if cv2.waitKey(1) & 0xFF == ord('p'):
                break

        count += 1
        if time.time() - sTime > 1:
            if cpsDisplay:
                print(f"CPS: {count}", end='\r')

            count = 0
            sTime = time.time()

    camera.stop()

def main_menu():
    while True:
        display_title()
        print("Escolha uma opção:")
        print("1. Editar configurações")
        print("2. Iniciar Aimbot")
        print("3. Sair")
        choice = input("Digite o número da opção: ")
        if choice == '1':
            config_data = load_user_config()
            descriptions = {
                "aaMovementAmp": "Multiplicador de movimento para aimbot (0.0 a 1.0).",
                "headshot_mode": "Ativar ou desativar o modo de cabeça (True/False).",
                "target_fps": "Frames por segundo alvo para o aimbot.",
                "onnxChoice": "Escolha de modelo ONNX, afeta a performance (1=CPU, 2=GPU AMD, 3=GPU NVIDIA).",
                "confidence": "Valor de confiança mínimo para disparar no alvo (0.0 a 1.0).",
                "cpsDisplay": "Exibir o CPS (Captures por Segundo) no HUD (True/False).",
                "aaQuitKey": "Tecla para sair do aimbot (ex: 'p')."
            }
            while edit_config(config_data, descriptions):
                pass
        elif choice == '2':
            config = load_user_config()
            global selected_model, aaMovementAmp, headshot_mode, targett_fps, onnxChoice, confidence, useMask, maskSide, maskWidth, maskHeight, cpsDisplay, visuals, centerOfScreen, aaQuitKey
            selected_model = config["selected_model"]
            aaMovementAmp = config["aaMovementAmp"]
            headshot_mode = config["headshot_mode"]
            targett_fps = config["target_fps"]
            onnxChoice = config["onnxChoice"]
            confidence = config["confidence"]
            useMask = config["useMask"]
            maskSide = config["maskSide"]
            maskWidth = config["maskWidth"]
            maskHeight = config["maskHeight"]
            cpsDisplay = config["cpsDisplay"]
            visuals = config["visuals"]
            centerOfScreen = config["centerOfScreen"]
            aaQuitKey = config["aaQuitKey"]
            main(targett_fps)
            input("Pressione Enter para voltar ao menu principal...")
        elif choice == '3':
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")
            time.sleep(2)


if __name__ == "__main__":
    try:
        main_menu()
    except Exception as e:
        import traceback
        traceback.print_exception(e)
        print("ERROR:", e)
