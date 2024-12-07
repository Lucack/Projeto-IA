
---

# Projeto IA - Agente Inteligente para Super Mario World

## Descrição do Projeto

Este projeto consiste na implementação de um agente inteligente capaz de jogar a fase Yoshilsland2 do jogo Super Mario World por meio da biblioteca Retro Gym e Neat.

## Integrantes do Grupo

- Gabriel Cobo Figueiro RA: 11202131397
- Kayque de Moraes Urbano Oliveira RA: 11202130076
- Lucas Santana Santos RA: 11202130726
- André Kenji Sato RA: 11201811058

## Resultado
[![Agente Inteligente jogando Super Mario World - Projeto IA
](https://img.youtube.com/vi/HZQTkN1S9kk/0.jpg)](https://www.youtube.com/watch?v=HZQTkN1S9kk)


## Objetivos

- Implementar um agente que consiga passar da fase Yoshilsland2.
- Fazer com que o agente mais rápido seja escolhido.
- Permitir o aprimoramento contínuo do agente.

## Tecnologias Utilizadas

- Linguagem de Programação: Python 3.8.0
- Biblioteca: Retro Gym (https://github.com/openai/retro)
- Biblioteca: NEAT (https://neat-python.readthedocs.io/en/latest/)

## Instalação e Configuração

### Configurar o Ambiente Python

1. Verifique a versão do Python instalada no seu sistema. O projeto requer **Python 3.8**:
   ```bash
   python --version
   ```
   Caso a versão instalada não seja a 3.8, faça o download em [Python.org](https://www.python.org/).

2. Verifique a versão do `pip`:
   ```bash
   pip --version
   ```
   É recomendado utilizar o **pip 19.2.3**.
   Caso sua versão seja superior, ou enfrente problemas, instale o pip recomendado: **python.exe -m pip install pip==19.2.3**


3. Crie um ambiente virtual para o projeto:
   ```bash
   python -m venv mario
   ```

4. Ative o ambiente virtual:
   - **Windows**:
     ```bash
     .\mario\Scripts\activate
     ```
   - **Linux/Mac**:
     ```bash
     source mario/bin/activate
     ```

### Instalar Dependências

1. Com o ambiente virtual ativado, instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

### Configurar a Biblioteca Retro Gym

1. Copie os arquivos do jogo para o diretório correto dentro do ambiente virtual:
   ```bash
   xcopy "arquivos\data.json" "mario\Lib\site-packages\retro\data\stable\SuperMarioWorld-Snes\" /Y

   xcopy "arquivos\scenario.json" "mario\Lib\site-packages\retro\data\stable\SuperMarioWorld-Snes\" /Y

   xcopy "arquivos\rom.sfc" "mario\Lib\site-packages\retro\data\stable\SuperMarioWorld-Snes\" /Y

   ```

## Como Executar

1. Para rodar o agente:
   ```bash
   python play.py
   ```

2. Para treinar o agente:
   ```bash
   python train.py
   ```

---
