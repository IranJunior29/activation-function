# Imports

import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

if __name__ == '__main__':

    # Função para transformar os dados ao carregar, aplicando normalização com média e desvio padrão de 0.5
    transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,))])

    # Carrega os dados de treino
    trainig_data = torch.utils.data.DataLoader(datasets.MNIST('dados',
                                                              train=True,
                                                              download=True,
                                                              transform=transforms),
                                                              batch_size=64,
                                                              shuffle=True)

    # Carrega os dados de teste
    test_data = torch.utils.data.DataLoader(datasets.MNIST('dados',
                                                      train=False,
                                                      transform=transforms),
                                                      batch_size=64,
                                                      shuffle=True)

    # Visualizando a dimensão dos dados de treino
    dataiter = iter(trainig_data)
    images, labels = next(dataiter)

    print(images.shape)
    print(labels.shape)

    # Visualizando uma imagem
    plt.imshow(images[0].numpy().squeeze(), cmap='gray_r');

    # Visualizando várias imagens
    figure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images + 1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

    # Hiperparâmetros da rede
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # Modelo com função de ativação LeakyReLU
    modelo_fa2 = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),  # 784 x 128
                          nn.LeakyReLU(0.2),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),  # 128 x 64
                          nn.LeakyReLU(0.2),
                          nn.Linear(hidden_sizes[1], output_size),  # 64 x 10
                          nn.LogSoftmax(dim = 1))

    print(modelo_fa2)

    # Função de custo do modelo
    criterion = nn.NLLLoss()

    # Obtendo os lotes de dados
    images, labels = next(iter(trainig_data))

    # Ajustando o shape
    images = images.view(images.shape[0], -1)

    # Logs das probabilidades de classe
    logps = modelo_fa2(images)

    # Perda
    loss = criterion(logps, labels)

    print('\nAntes do Backward Pass: \n', modelo_fa2[0].weight.grad)
    loss.backward()
    print('\nDepois do Backward Pass: \n', modelo_fa2[0].weight.grad)

    # Otimizador
    optimizer = optim.SGD(modelo_fa2.parameters(), lr=0.003, momentum=0.9)

    # Número de épocas
    epochs = 5

    # Loop de treinamento
    for e in range(epochs):

        # Registra o momento de início da época
        start_time = time.time()

        # Zera o erro da época
        running_loss = 0

        # Loop pelas imagens e labels
        for images, labels in trainig_data:

            # Flatten das imagens
            images = images.view(images.shape[0], -1)

            # Zera os gradientes
            optimizer.zero_grad()

            # Previsão com o modelo
            output = modelo_fa2(images)

            # Cálculo do erro
            loss = criterion(output, labels)

            # Aqui acontece o aprendizado com backpropagation
            loss.backward()

            # E aqui otimiza os pesos
            optimizer.step()

            running_loss += loss.item()
        else:
            print('Epoch: {0}, Tempo Decorrido: {1:.2f}s, Loss(Erro): {2}'.format(e,
                                                                                  time.time() - start_time,
                                                                                  running_loss / len(trainig_data)))


    # Função para visualizar a classificação
    def visualiza_classe(img, ps):
        ps = ps.data.numpy().squeeze()
        fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
        ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
        ax1.axis('off')
        ax2.barh(np.arange(10), ps)
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(10))
        ax2.set_yticklabels(np.arange(10))
        ax2.set_title('Probabilidade de Classe')
        ax2.set_xlim(0, 1.1)
        plt.tight_layout()


    # Obtém uma imagem de teste
    images, labels = next(iter(test_data))

    # Ajusta a imagem
    img = images[0].view(1, 784)

    # Faz previsão com a imagem
    with torch.no_grad():
        logps = modelo_fa2(img)

    # Log de probabilidade da previsão
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])

    # Print
    print("Dígito Previsto =", probab.index(max(probab)))

    # Visualiza classe
    visualiza_classe(img.view(1, 28, 28), ps)

    # Avaliando o modelo

    # Contadores
    correct_count, all_count = 0, 0

    # Loop
    for images, labels in test_data:
        for i in range(len(labels)):

            img = images[i].view(1, 784)

            with torch.no_grad():
                logps = modelo_fa2(img)

            ps = torch.exp(logps)

            probab = list(ps.numpy()[0])

            pred_label = probab.index(max(probab))

            true_label = labels.numpy()[i]

            if (true_label == pred_label):
                correct_count += 1

            all_count += 1

    print("Número de Imagens Testadas =", all_count)
    print("Acurácia nos Dados de Teste (%) =", (correct_count / all_count) * 100)

    # Salvando o modelo
    torch.save(modelo_fa2, 'modelos/modelo_fa2.pt')


