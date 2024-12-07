import retro
import os


current_directory = os.getcwd()
print(f"O diretório atual de execução é: {current_directory}")


env = retro.make('SuperMarioWorld-Snes', 'YoshiIsland2', players=1)

env.reset()

done = False

while not done:

    env.render()

    # action = env.action_space.sample()

    # print(action)
    action = [130]

    # action = [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]

    # jump = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0] 
    # down = [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1]

    ob, reward , done , info = env.step(action) # every frame
    print("Action ", action , " | Reward: ", reward , " | Done", done, "\nInfo", info, "\n")

