# Venus Lite

![Venus](https://i.imgur.com/X3M2USt.gif)

Venus Lite é um aimbot baseado em IA focado em Fortnite mas agora com modelos universais, utilizando **YOLO v5** para detecção de inimigos em tempo real. Ele captura a tela, processa os frames com YOLO e move o mouse automaticamente para mirar no alvo.

## Características
✅ Modelos Universais de AI para praticamente qualquer jogo de tiro<br>
✅ Detecção de inimigos em tempo real com YOLO v5  
✅ Suporte para GPUs NVIDIA e AMD  
✅ Algoritmo de tracking preciso  
✅ Ajuste de suavidade e delay para evitar detecção  
✅ Interface de configuração futura<br>
✅ Bibloteca indetectável pelo jogo

---

## 🔧 Requisitos

### 🖥️ Hardware:
- **GPU NVIDIA** (CUDA) ou **GPU AMD** (OpenCL)  
- Processador razoável para rodar inferência da IA  
- Windows 10/11  

### 📦 Softwares & Dependências:
- Python 3.11.6  
- PyTorch e Torchvision  
- OpenCV  
- mss (para captura de tela)  
- pynput (para controle do mouse)  
- win32api e win32con (para emulação de entrada indetectável)  
- YOLO v5

---

## 🚀 Instalação

1. **Clone o repositório**
   ```sh
   git clone https://github.com/Merctxt/Venus-lite.git
   cd VenusAimBot
   ```

2. **Crie um ambiente virtual (opcional, mas recomendado)**
   ```sh
   python -m venv venv
   source venv/bin/activate  # Linux
   venv\Scripts\activate  # Windows
   ```

3. **Instale as dependências**
   ```sh
   pip install -r requirements.txt
   ```

4. **Execute o script**
   ```sh
   python src.py
   ```

---

## 🎮 Como Usar
- Inicie o *Fortnite* ou o jogo de sua escolha em modo janela sem bordas
- Ajuste os parâmetros no arquivo de configuração
- Pressione a tecla atribuída para ativar o aimbot
- O bot irá detectar inimigos e mover a mira automaticamente

---

## ⚠️ Aviso Legal
**Este software é apenas para fins educacionais.** Por mais que o Venus Lite seja atualmente indetectável o uso de aimbots em jogos online pode resultar em banimento da conta. Use por sua conta e risco.

---

## 📌 Roadmap Futuro
- [ ] Melhorar a detecção com modelos mais leves
- [ ] Adicionar interface gráfica (UI) para configuração
- [✅] Suporte para mais jogos

Se tiver dúvidas ou sugestões, contribua ou abra uma issue! 🚀

