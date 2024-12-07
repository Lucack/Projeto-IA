import retro
import numpy as np
import cv2
import neat
import pickle
import time

# Inicializa o ambiente do jogo Sonic
env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland2', players=1)

# Função para rodar o ganhador salvo
def rodar_ganhador(genoma, config):
    """
    Função para rodar o genoma vencedor (carregado de 'ganhador.pkl') no ambiente.
    """
    # Resetando o ambiente e pegando a observação inicial
    observador = env.reset()

    # Obtendo as dimensões da observação
    inputX, inputY, inputColors = env.observation_space.shape

    # Criando a rede neural a partir do genoma e da configuração
    rede = neat.nn.recurrent.RecurrentNetwork.create(genoma, config)

    cabou = False  # Condição de término do jogo
    fitness_atual = 0  # Fitness atual
    frame = 0  # Contador de frames

    frame_duration = 1 / 60

    while not cabou:
        start_time = time.time()
        env.render()  # Renderiza o jogo (mostrar na tela)
        frame += 1  # Incrementa o contador de frames
        print("FRAMEE ", frame)

        # Redimensiona a imagem para 13x13 para a rede neural (a entrada que a rede espera)
        observador = cv2.resize(observador, (13, 13))  # Redimensionando para 13x13
        observador = cv2.cvtColor(observador, cv2.COLOR_BGR2GRAY)  # Convertendo para escala de cinza
        observador = np.reshape(observador, (13 * 13,))  # Flattening da imagem para um vetor 1D (169 entradas)

        # Passa a imagem para a rede neural
        nnOutput = rede.activate(observador)

        # Define o vetor de ações específicas
        acoes_especificas = [131, 384, 130, 386]

        # Inicializa o vetor de ação com zeros
        action = np.zeros(len(acoes_especificas))  # O tamanho do vetor de ações será o tamanho de acoes_especificas

        # Seleciona a ação com maior valor de nnOutput
        indice_acao = np.argmax(nnOutput)  # Encontra o índice da maior saída da rede

        # A ação selecionada será aquela no índice correspondente no vetor de ações
        action[indice_acao] = acoes_especificas[indice_acao]

        # Executa a ação no ambiente
        observador, recompensa, done, info = env.step(action)

        # Se o jogo acabou, encerra o loop
        if done:
            cabou = True

        # Acumula a recompensa para calcular o fitness
        fitness_atual += recompensa

        # Exibe o fitness atual
        print(f"Fitness Atual: {fitness_atual}")

        elapsed_time = time.time() - start_time
        time_to_sleep = max(0, frame_duration - elapsed_time)
        time.sleep(time_to_sleep)

    print(f"Jogo Finalizado! Fitness Final: {fitness_atual}")

# Carregando o melhor genoma (ganhador.pkl)
with open('checkpointIGUAL\ganhadorIGUAL.pkl', 'rb') as f:
    ganhador = pickle.load(f)

# Configuração do NEAT
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

# Rodando o melhor genoma no ambiente
rodar_ganhador(ganhador, config)
