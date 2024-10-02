# Projeto IA - Agente Inteligente para Super Mario World

## Descrição do Projeto

Este projeto consiste na implementação de um agente inteligente capaz de jogar a fase Yoshilsland2 do jogo Super Mario World. O agente será desenvolvido utilizando algoritmos em Python e fará uso da biblioteca Retro Gym, conforme as diretrizes apresentadas nas aulas do curso.

## Integrantes do Grupo

- Gabriel Cobo Figueiro RA: 11202131397
- Kayque de Moraes Urbano Oliveira RA: 11202130076
- Lucas Santana Santos RA: 11202130726
- Paulo Victor Dias Soares RA: 11202111146
- Ruhama do Nascimento Ciriaco Pereira RA: 11202111584
- Vinicius de Camargo Vieira RA: 11202130791

### Código para Carregar o Jogo

Para carregar o jogo, utilize a seguinte linha de código:

```python
env = retro.make(game='Super Mario World-Snes', state='Yoshilsland2', players=1)
```

## Objetivos

- Implementar um agente que consiga navegar pela fase Yoshilsland2.
- Aplicar um algoritmo de inteligência artificial adequado para o desafio.
- Aprender na prática sobre os conceitos de agentes inteligentes e técnicas de IA.

## Tecnologias Utilizadas

- Linguagem de Programação: Python
- Biblioteca: Retro Gym (https://github.com/openai/retro)

## Instalação

Para instalar a biblioteca Retro Gym, siga as instruções disponíveis em: [Retro Gym - GitHub](https://github.com/openai/retro).

Além disso, copie a ROM do jogo para o diretório `site-packages/retro/data/stable/Super Mario WorldSnes/` com o nome `rom.sfc`. Se estiver utilizando Anaconda, o caminho pode ser semelhante a `~/anaconda3/lib/python3.11/`.

## Como Executar

Para rodar o agente, utilize o seguinte comando:

```bash
python main.py
```

---
