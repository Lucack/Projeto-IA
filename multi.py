import retro
import numpy as np
import cv2
import neat
import pickle
import os
from concurrent.futures import ThreadPoolExecutor

# Inicializa o ambiente do jogo Sonic
env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland2', players=1)

# Variáveis auxiliares
imgarray = []
xpos_end = 0

def avaliarGenoma(args):
    """
    Avalia um único genoma.
    Args:
        args: Tuple contendo (genoma_id, genoma, config)
    """
    genoma_id, genoma, config = args
    observador = env.reset()
    acao = env.action_space.sample()

    # Dimensões de entrada ajustadas
    inputX, inputY = 13, 13

    # Criando a rede neural a partir do genoma
    rede = neat.nn.recurrent.RecurrentNetwork.create(genoma, config)

    max_fitness_atual = 0  # Fitness máximo atual
    fitness_atual = 0  # Fitness atual
    frame = 0  # Contador de frames
    contador = 0  # Contador para estagnação
    xpos = 0  # Posição no jogo
    xpos_end = 0  # Posição final do nível
    cabou = False
    scoreAdicional = False

    while not cabou:
        # env.render()
        frame += 1
        observador = cv2.resize(observador, (inputX, inputY))
        observador = cv2.cvtColor(observador, cv2.COLOR_BGR2GRAY)
        observador = np.reshape(observador, (inputX * inputY,))

       # Passa a imagem para a rede neural
        nnOutput = rede.activate(observador)

        # Define o vetor de ações específicas
        acoes_especificas = [131, 384, 130, 386]

        # Inicializa o vetor de ação com zeros
        action = np.zeros(len(acoes_especificas))  # O tamanho do vetor de ações será o tamanho de acoes_especificas

        # Seleciona a ação com maior valor de nnOutput
        # A ideia aqui é mapear o maior valor de nnOutput para o vetor de ações específicas
        indice_acao = np.argmax(nnOutput)  # Encontra o índice da maior saída da rede

        # A ação selecionada será aquela no índice correspondente no vetor de ações
        action[indice_acao] = acoes_especificas[indice_acao]

        # Executa a ação no ambiente
        observador, recompensa, done, info = env.step(action)
        xpos = info['xpos']
        xpos_end = info['endOfLevel_x']

        fitness_atual += recompensa

        if fitness_atual > max_fitness_atual:
            max_fitness_atual = fitness_atual
            contador = 0
        else:
            contador += 1

        if done or contador == 250:
            cabou = True

        if xpos >= xpos_end and not scoreAdicional:
            print("Adicionando ao fitness",(3000-frame))

            fitness_atual += (3000 - frame)
            scoreAdicional = True

        genoma.fitness = fitness_atual

    print(f"Genoma {genoma_id} - Fitness: {fitness_atual}\n")

def avaliarGenomas(genomas, config):
    """
    Avalia os genomas utilizando multithreading.
    """
    args = [(genoma_id, genoma, config) for genoma_id, genoma in genomas]
    with ThreadPoolExecutor() as executor:
        executor.map(avaliarGenoma, args)

# Configuração do NEAT
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforwardNova')

# Verifica se há checkpoints existentes
checkpoint_dir = 'MultCheckpoint'
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint-multi-')]

if checkpoint_files:
    checkpoint_file = sorted(checkpoint_files, key=lambda x: int(x.split('-')[-1]))[-1]
    print(f"Carregando checkpoint {checkpoint_file}...")
    populacao = neat.Checkpointer.restore_checkpoint(os.path.join(checkpoint_dir, checkpoint_file))
else:
    populacao = neat.Population(config)

# Adicionando repórteres
populacao.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
populacao.add_reporter(stats)
populacao.add_reporter(neat.Checkpointer(10, filename_prefix="checkpoint-multi-"))

# Executando a evolução
ganhador = populacao.run(avaliarGenomas, 50)

# Salvando o melhor genoma
with open('MultCheckpoint/ganhadorMulti.pkl', 'wb') as output:
    pickle.dump(ganhador, output, 1)
