import retro
import numpy as np
import cv2
import neat
import pickle
import os

# Inicializa o ambiente do jogo Sonic
env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland2', players=1)

# Variáveis auxiliares
imgarray = []
xpos_end = 0

def avaliarGenomas(genomas, config):
    """
    Função de avaliação dos genomas: avalia o desempenho de cada genoma no ambiente.
    """
    for genoma_id, genoma in genomas:
        # Resetando o ambiente e pegando a observação inicial
        observador = env.reset()
        acao = env.action_space.sample()

        # Obtendo as dimensões da observação
        inputX, inputY, inputColors = env.observation_space.shape

        # Ajustando as dimensões para 13x13 para a rede neural
        inputX = 13  
        inputY = 13  

        # Criando a rede neural a partir do genoma e da configuração
        rede = neat.nn.recurrent.RecurrentNetwork.create(genoma, config)

        max_fitness_atual = 0  # Fitness máximo atual
        fitness_atual = 0  # Fitness atual
        frame = 0  # Contador de frames
        contador = 0  # Contador para a condição de estagnação
        xpos = 0  # Posição do Sonic no jogo
        xpos_max = 0  # Posição máxima alcançada

        cabou = False  # Condição de término do jogo
        scoreAdicional = False

        while not cabou:
            env.render()  # Renderiza o jogo (mostrar na tela)
            frame += 1  # Incrementa o contador de frames

            # Redimensiona a imagem para 13x13
            observador = cv2.resize(observador, (inputX, inputY))
            observador = cv2.cvtColor(observador, cv2.COLOR_BGR2GRAY)  # Convertendo para escala de cinza
            observador = np.reshape(observador, (inputX * inputY,))  # Flattening da imagem para um vetor 1D

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
            xpos = info['xpos']  # Posição do personagem
            game_score = info['score']  # Pontuação do jogo
            lives = info['lives']  # Número de vidas restantes
            end_of_level = info['endOfLevel']  # Posição final do nível
            xpos_end = info['endOfLevel_x']  # Posição final do nível
            coins = info['coins']  # Número de moedas coletadas

            # Se o jogo acabou, encerra o loop
            if done:
                cabou = True

            # Acumula a recompensa para calcular o fitness
            fitness_atual += recompensa

            # Condição para atualizar o fitness máximo
            if fitness_atual > max_fitness_atual:
                max_fitness_atual = fitness_atual
                contador = 0  # Resetando o contador se houve melhoria no fitness
            else:
                contador += 1  # Incrementa o contador de estagnação

            # Se o jogo acabou ou se o contador atingiu o limite (250), encerra
            if done or contador == 250:
                cabou = True
                print(f"Genoma {genoma_id} - Fitness: {fitness_atual}")

            if (xpos >= xpos_end and scoreAdicional == False):
                print("Adicionando ao fitness",(3000-frame))
                fitness_atual += (3000-frame)
                scoreAdicional = True

            # Atualiza o fitness do genoma
            genoma.fitness = fitness_atual

# Configuração do NEAT
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

# Verifica se há checkpoints existentes
checkpoint_dir = 'checkpointIGUAL'  # Ou onde seus arquivos de checkpoint estão armazenados
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint-IGUAL-')]

if checkpoint_files:
    # Carregar o checkpoint mais recente
    checkpoint_file = sorted(checkpoint_files, key=lambda x: int(x.split('-')[-1]))[-1]
    print(f"Carregando checkpoint {checkpoint_file}...")
    populacao = neat.Checkpointer.restore_checkpoint(os.path.join(checkpoint_dir, checkpoint_file))
else:
    # Caso não haja checkpoint, inicializa a população do zero
    populacao = neat.Population(config)

# Adicionando os repórteres para o NEAT
populacao.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
populacao.add_reporter(stats)
populacao.add_reporter(neat.Checkpointer(10,filename_prefix="checkpoint-IGUAL-"))

# Executando a evolução
ganhador = populacao.run(avaliarGenomas, 50)

# Salvando o melhor genoma
with open('checkpointIGUAL\ganhadorIGUAL.pkl', 'wb') as output:
    pickle.dump(ganhador, output, 1)
