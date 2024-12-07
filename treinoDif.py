import retro
import numpy as np
import cv2
import neat
import pickle
import os
from multiprocessing import Pool

# Inicializa o ambiente do jogo Super Mario World
env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland2', players=1)

# Caminho para os checkpoints
checkpoint_dir = 'Save'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Variáveis auxiliares
prev_action = None  # Para penalizar ações repetitivas

def avaliarGenoma(genoma_config):
    genoma_id, genoma, config = genoma_config
    observador = env.reset()
    acao = env.action_space.sample()
    inputX, inputY, _ = env.observation_space.shape

    inputX, inputY = 13, 13

    rede = neat.nn.recurrent.RecurrentNetwork.create(genoma, config)

    fitness_atual = 0
    max_fitness_atual = 0
    contador = 0
    xpos_max = 0
    frame = 0
    cabou = False

    global prev_action
    prev_action = np.zeros(env.action_space.n)

    while not cabou:
        # env.render()
        frame += 1

        observador = cv2.resize(observador, (inputX, inputY))
        observador = cv2.cvtColor(observador, cv2.COLOR_BGR2GRAY)
        observador = observador / 255.0
        observador = np.reshape(observador, (inputX * inputY,))

        nnOutput = rede.activate(observador)
        action = np.zeros(env.action_space.n)
        action[np.argmax(nnOutput)] = 1

        observador, recompensa, done, info = env.step(action)
        xpos = info['xpos']
        xpos_end = info['endOfLevel_x']

        # Atualizar fitness
        fitness_atual += recompensa + (xpos / xpos_end) * 1000
        if np.array_equal(action, prev_action):
            fitness_atual -= 0.1
        prev_action = action

        # Verificar progresso
        if xpos > xpos_max:
            xpos_max = xpos
            contador = 0  # Reset contador se progresso ocorrer
        else:
            contador += 1
            fitness_atual -= 1  # Penalidade por frame sem progresso

        # Condições de término
        if done or contador >= 250 or frame >= 5000:
            cabou = True

    genoma.fitness = fitness_atual
    return genoma_id, fitness_atual

def avaliarGenomas(genomas, config):
    """
    Avalia uma população de genomas em paralelo.
    """
    pool = Pool()  # Cria pool de processos
    resultados = pool.map(avaliarGenoma, [(genoma_id, genoma, config) for genoma_id, genoma in genomas])
    pool.close()
    pool.join()

    # Atualiza fitness na população
    for genoma_id, fitness in resultados:
        for gid, g in genomas:
            if gid == genoma_id:
                g.fitness = fitness

if __name__ == '__main__':
    # Configuração do NEAT
    config = neat.Config(
        neat.DefaultGenome, 
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet, 
        neat.DefaultStagnation,
        'config-feedforwardNova'
    )

    # Verifica se há checkpoints existentes
    checkpoint_dir = 'Save'
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint-save-')
    ]

    if checkpoint_files:
    # Carregar o checkpoint mais recente
        try:
            checkpoint_file = sorted(checkpoint_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))[-1]
            print(f"Carregando checkpoint {checkpoint_file}...")
            populacao = neat.Checkpointer.restore_checkpoint(os.path.join(checkpoint_dir, checkpoint_file))
        except Exception as e:
            print(f"Erro ao carregar checkpoint: {e}")
            print("Inicializando nova população...")
            populacao = neat.Population(config)
    else:
        # Caso não haja checkpoint, inicializa a população do zero
        print("Nenhum checkpoint encontrado. Inicializando nova população...")
        populacao = neat.Population(config)

    # Adicionando os repórteres para o NEAT
    populacao.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    populacao.add_reporter(stats)
    populacao.add_reporter(neat.Checkpointer(10,filename_prefix="checkpoint-save-"))

    # Executando a evolução
    ganhador = populacao.run(avaliarGenomas, 50)

    # Salvando o melhor genoma
    with open('ganhador.pkl', 'wb') as output:
        pickle.dump(ganhador, output, 1)