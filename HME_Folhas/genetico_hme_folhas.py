import numpy as np
import random 
from operator import attrgetter
import math as math
import time as time
from copy import deepcopy
from enum import Enum
from oct2py import octave
from oct2py import Oct2Py
from threading import Thread
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
# FUNCIONAR TOTALMENTE, INSTALAR O PYTHON E AS SEGUINTES BIBLIOTECAS
# pip install plotly
# pip install oct2py
# pip install pickle
# ESPECIFICAR AQUI O CAMINHO ONDE ESTÃO OS ARQUIVOS OCTAVE/MATLAB
octave.addpath("C:/Users/miche/Desktop/TRABALHO FINAL/Octave_HME_Folhas")
# NECESSÁRIO TER O OCTAVE INSTALADO NA MÁQUINA, COM A SEGUINTE VARIAVEL DE AMBIENTE:
# PATH  => C:\Octave\Octave-5.1.0.0\mingw64\bin
# OCTAVE_EXECUTABLE => C:\Octave\Octave-5.1.0.0\mingw64\bin\octave-cli.exe

class TIPO_SELECAO(Enum):
    ROLETA = 1
    AMOSTRAGEM_ESTOCASTICA = 2 
    TORNEIO = 3
    RANK_LINEAR = 4

class TIPO_CROSSOVER(Enum):
    UNICO_PONTO = 1
    DUPLO_PONTO = 2 
    UNIFORME = 3
    ARITMETICO = 4

class TIPO_MUTACAO(Enum):
    GENE_UNICO = 1 # Se houver, escolhe 1 gene e efetua a mutação
    N_GENES = 2 # Se houver, escolhe N genes e efetua a mutação
    TODOS_GENES = 3 # Percorre cada gene e, se houver mutaçãp, efetua a mutação neste gene

class Individuo():
    def __init__(self, bits_altura, bits_largura, bits_folhas, geracao=0):
        self.cromossomo = np.random.randint(2, size=bits_altura+bits_largura+bits_folhas) #altura = 2bits / largura=3 bits
        self.bits_largura = bits_largura
        self.bits_altura = bits_altura
        self.bits_folhas = bits_folhas
        self.fitness = -10000
        self.geracao = geracao
        self.qtde_bits_por_entrada = bits_altura + bits_largura + bits_folhas
        self.atualizar()

    def adicionar_geracao(self):
        self.geracao = self.geracao + 1
        
    def atualizar(self):
        self.x = self.cromossomo[0:self.bits_altura] 
        self.y = self.cromossomo[self.bits_altura:self.bits_altura+self.bits_largura] 
        self.folhas = self.cromossomo[self.bits_largura+self.bits_altura::]

    def getProfundidade(self):
        return self.cromossomo[0:self.bits_altura] # 0 a 2 (altura) + 1

    def getRamos(self):
        return self.cromossomo[self.bits_altura:self.bits_largura+self.bits_altura] # 2 a 5 (largura)
    
    def getFolhas(self):
        return self.cromossomo[self.bits_largura+self.bits_altura::] # 2 a 5 (largura)
    
    def atualizarXY(self):
        self.x = self.cromossomo[0:self.bits_altura] 
        self.y = self.cromossomo[self.bits_altura:self.bits_largura+self.bits_altura]
        self.folhas = self.cromossomo[self.bits_largura+self.bits_altura::]

    def setXY(self, cromo):
        self.cromossomo = cromo
        if(self.geracao == 0):
            self.geracao = 1
        
        self.atualizar()

class AlgoritmoGenetico():

    

    def __init__(self, 
                    bits_altura, 
                    bits_largura, 
                    bits_folha_unica,
                    geracoes, 
                    individuos_por_geracao, 
                    qtde_bits_por_entrada, 
                    tipo_selecao=TIPO_SELECAO.ROLETA, # Qual o tipo de seleção aplicada
                    qtde_individuo_elitismo=0, 
                    crossover_por_entrada=False, # True ou False. Indica se o crossover será feito por entrada (ex.: POR ENTRADA => pai1.x com pai2.x e pai1.y com pai2.y) ou no cromossomo inteiro (ex.: INTEIRO => pai1.cromossomo com pai2.cromossomo)
                    tipo_crossover=TIPO_CROSSOVER.UNICO_PONTO,
                    taxa_crossover=0.8,
                    taxa_selecao_melhor_torneio=0.8, 
                    tipo_mutacao=TIPO_MUTACAO.GENE_UNICO,
                    taxa_mutacao=0.01,
                    populacao_inicial = []):
        
        global combinacoes_resultados
        global combinacoes_yexp
        global combinacoes_z
        combinacoes_resultados = {}
        combinacoes_yexp = {}
        combinacoes_z = {}

        self.bits_altura = bits_altura
        self.bits_largura = bits_largura
        max_altura = int(self.calcular_valor_inteiro(np.ones(bits_altura), True))
        max_largura = int(self.calcular_valor_inteiro(np.ones(bits_largura), False))
        self.bits_folhas = max_largura**(max_altura-1)
        self.bits_folhas = bits_folha_unica * self.bits_folhas
        self.qtde_bits = self.bits_altura + self.bits_largura + self.bits_folhas
        self.bits_folha_unica = bits_folha_unica
        
        self.individuos_por_geracao = individuos_por_geracao
        self.tipo_selecao = tipo_selecao
        self.qtde_individuo_elitismo = qtde_individuo_elitismo
        self.crossover_por_entrada = crossover_por_entrada
        self.tipo_crossover = tipo_crossover
        self.taxa_crossover = taxa_crossover
        self.geracoes = geracoes
        
        self.numero_geracao = 1
        self.tipo_mutacao = tipo_mutacao
        self.taxa_mutacao = taxa_mutacao
        self.taxa_selecao_melhor = taxa_selecao_melhor_torneio
        self.fitnesses = []
        self.piores_individuos = []
        self.melhores_individuos = []
        self.media_fitnesses = []
        self.populacao_inicial = populacao_inicial
        self.populacao = [] 
        self.melhor_individuo = Individuo(bits_altura, bits_largura, self.bits_folhas)

    def calcular_valor_inteiro(self, binario, altura=False):
        valor = 0
        exp = 0
        for i in reversed(range(len(binario))):
            valor = valor + (binario[i] * 2**exp)
            exp = exp + 1
        if(altura):
            valor = valor + 1
            # if(valor == 0):
            #     valor = 1
        else:
            valor = valor + 2 # Se for largura/ramo, adiciona 2, pois é o minimo de ramo neste algoritmo
        return valor

    # Gera a população inicial, de forma aleatória, com a quantidade de individuos pré-definida
    def gerar_populacao_inicial(self):
        if len(self.populacao_inicial) <= 0:
            for i in range(self.individuos_por_geracao):
                self.populacao.append(Individuo(self.bits_altura, self.bits_largura, self.bits_folhas, self.numero_geracao))
        else:
            for i in range(self.individuos_por_geracao):
                self.populacao.append(self.populacao_inicial[i])
    
    # Avalia a população, gerando o fitness
    def avaliar_populacao(self):
        MAX_THREADS = mp.cpu_count()
        thread_pool = ThreadPool(processes=MAX_THREADS)
        for i in range(self.individuos_por_geracao):
            thread_pool.apply_async(self.calcular_funcao, args=([self.populacao[i],i]))
        thread_pool.close() # conclui a adição das threads no ppol
        # thread_pool.terminate()
        thread_pool.join() # Espera todas as threads adicionadas acima terminarem


    # Calculo da função que se deseja minimizar
    def calcular_funcao(self, individuo, i):
        
        altura = self.calcular_valor_inteiro(individuo.getProfundidade(), True) # altura deve ser no minimo 1
        ramos = self.calcular_valor_inteiro(individuo.getRamos()) # ramo deve ser no minimo 2
        folhas = []
        folhas_individuo = individuo.getFolhas()
        for j in range(0, len(folhas_individuo), self.bits_folha_unica):
            valor = self.calcular_valor_inteiro(folhas_individuo[j:j+self.bits_folha_unica])
            folhas.append(valor)

        max_folhas_ind = ramos ** (altura-1) # pega somente a quantidade de folhas q a estrutura terá
        folhas = folhas[0:max_folhas_ind]
        folhas = np.array(folhas)
        print("\nGer" + str(individuo.geracao) + "_Ind" + str(i) + "(" + str(altura) + "," + str(ramos) + ") -> " + str(folhas))

        chave = str(altura) + "," + str(ramos) + "," + str(folhas)
        if chave in combinacoes_resultados.keys():
            lli = combinacoes_resultados[chave]
            yexp = combinacoes_yexp[chave]
            z = combinacoes_z[chave]
        else:
            octave = Oct2Py()
            hme_fit, yexp, z = octave.hmeTest(altura, ramos, folhas, nout=3)
            lli = hme_fit.lli
            if chave in combinacoes_resultados.keys():
                if combinacoes_resultados[chave] < lli: # for maior (ou seja, melhor), atualiza
                    combinacoes_resultados[chave] = lli
                    combinacoes_yexp[chave] = yexp
                    combinacoes_z[chave] = z
                else:
                    lli = combinacoes_resultados[chave]
                    yexp = combinacoes_yexp[chave]
                    z = combinacoes_z[chave]
            else:
                combinacoes_resultados[chave] = lli
                combinacoes_yexp[chave] = yexp
                combinacoes_z[chave] = z

            # octave.close()
            octave.exit()
        individuo.fitness = lli
        individuo.yexp = yexp
        individuo.z = z
        return lli 

    # Ordena a lista de Individuos de acordo com o fitness, do maior para o menor
    def ordenar_populacao(self):
        self.populacao = sorted(self.populacao,
                                key = lambda populacao: populacao.fitness,
                                reverse = True)
        
        self.fitnesses.append(self.populacao[0].fitness)
        self.piores_individuos.append(self.populacao[-1])
        self.melhores_individuos.append(self.populacao[0])

        media = np.mean([c.fitness for c in self.populacao])
        self.media_fitnesses.append(media)
        if self.populacao[0].fitness > self.melhor_individuo.fitness:
            self.melhor_individuo = self.populacao[0]
        
    # Imprime o melhor e pior indivíduo da geração
    def imprime_melhor_pior_da_geracao(self):
        melhor_fitness = self.populacao[0].fitness
        pior_fitness = self.populacao[-1].fitness
        x = self.calcular_valor_inteiro(self.populacao[0].getProfundidade(), True)
        y = self.calcular_valor_inteiro(self.populacao[0].getRamos()) + 2
        x_pior = self.calcular_valor_inteiro(self.populacao[-1].getProfundidade(), True)
        y_pior = self.calcular_valor_inteiro(self.populacao[-1].getRamos()) + 2
        
        # if (self.numero_geracao >= self.geracoes):
        print("\nGERAÇÃO %s" % (self.numero_geracao))
        print("Melhor: Fitness: %s Altura: %s Ramos: %s" % (melhor_fitness,
                                                            x,
                                                            y))
        print("Pior: Fitness: %s Altura: %s Ramos: %s" % (pior_fitness,
                                                               x_pior,
                                                               y_pior))                                                               
        print(combinacoes_resultados)

    # soma do fitness para utilizar na selção por roleta/amostragem estática
    def soma_total_fitness(self):
        soma = 0
        valor_ajuste = 0
        # Caso tenha numero negativo, ajusta para a soma ser a partir do 1, que para que seja possível efetuar a Roleta
        # É somado +1 para que, mesmo o último colocado, tenha chance de ser selecionado na roleta (apesar se ser pequena).
        # Resumindo, estamos ajustando o eixo dos valores para a partir de 0. Isso só é utilizado se a Roleta for utilizada para seleção
        if(self.populacao[-1] < 0):
            valor_ajuste = -self.populacao[-1].fitness + 1
        
        for individuo in self.populacao:
            soma += individuo.fitness + valor_ajuste
        return soma


    # Seleciona os pais para o crossover
    def selecionar_pais(self):
        if self.tipo_selecao == TIPO_SELECAO.ROLETA:
            return self.seleciona_pai_roleta()
        elif self.tipo_selecao == TIPO_SELECAO.AMOSTRAGEM_ESTOCASTICA:
            return self.seleciona_pai_amostragem_estocastica()
        elif self.tipo_selecao == TIPO_SELECAO.TORNEIO:
            return self.seleciona_pai_torneio()
        elif self.tipo_selecao == TIPO_SELECAO.RANK_LINEAR:
            return self.seleciona_pai_rank_linear()

    def calcular_soma_fitness(self, populacao_soma):
        soma_avaliacao = 0
        valor_ajuste = 0
        # Caso tenha numero negativo, ajusta para a soma ser a partir do 1, que para que seja possível efetuar a Roleta
        # É somado +1 para que, mesmo o último colocado, tenha chance de ser selecionado na roleta (apesar se ser pequena).
        # Resumindo, estamos ajustando o eixo dos valores para a partir de 0. Isso só é utilizado se a Roleta for utilizada para seleção
        if(populacao_soma[-1].fitness < 0):
            valor_ajuste = -populacao_soma[-1].fitness + 1
        
        for individuo in populacao_soma:
            soma_avaliacao += individuo.fitness + valor_ajuste
        
        return soma_avaliacao, valor_ajuste

    # Efetua seleção dos pais via Roleta
    def seleciona_pai_roleta(self):
        soma_avaliacao, valor_ajuste = self.calcular_soma_fitness(self.populacao)

        # Seleciona o PAI 1
        pai1 = -1
        valor_sorteado = random.random() * soma_avaliacao
        soma = 0
        i = 0
        while i < len(self.populacao) and soma < valor_sorteado:
            soma += self.populacao[i].fitness + valor_ajuste
            pai1 += 1
            i += 1
        
        # Seleciona o PAI 2
        pai2 = -1
        valor_sorteado = random.random() * soma_avaliacao
        soma = 0
        i = 0
        while i < len(self.populacao) and soma < valor_sorteado:
            soma += self.populacao[i].fitness + valor_ajuste
            pai2 += 1
            i += 1

        # Retorna os pais selecionados pela roleta
        return self.populacao[pai1], self.populacao[pai2] 


    def seleciona_pai_amostragem_estocastica(self):
        soma_avaliacao, valor_ajuste = self.calcular_soma_fitness(self.populacao)
        n = 2 # Quantidade de individuos selecionados
        # Calculo a média
        media = soma_avaliacao / n
        alfa = random.random() 
        delta = alfa * media

        populacao_embaralhada = deepcopy(self.populacao)
        random.shuffle(populacao_embaralhada)
        
        i = 0
        j = 0
        soma = populacao_embaralhada[0].fitness + valor_ajuste
        index_pais = []
        while i < n:
            if (delta < soma):
                index_pais.append(j)
                delta = delta + media
                i = i + 1
            else:
                j = j + 1
                soma = soma + populacao_embaralhada[j].fitness + valor_ajuste
        
        pai1 = populacao_embaralhada[index_pais[0]]
        pai2 = populacao_embaralhada[index_pais[1]]

        return pai1, pai2
            

    # Como existem fitness negativos, o Torneio pode ser realizado.
    # A seleção normal, que usa a somatória, não pode ser usada sem ter o fitness tratado(faço isso). O torneio pode.
    def seleciona_pai_torneio(self):
        # Selecionando PAI 1
        # sorteia 2 individuos da população
        pais_1 = random.sample(self.populacao, 2)
        r = random.uniform(0, 1)

        # Se r for menor que a taxa de seleção, o melhor fitness é selecionado       
        if r <= self.taxa_selecao_melhor:
            pai1 = max(pais_1, key=attrgetter('fitness'))
        else:
            pai1 = min(pais_1, key=attrgetter('fitness'))

        ####################################################

        # Selecionando PAI 2
        # sorteia 2 individuos da população
        pais_2 = random.sample(self.populacao, 2)
        r = random.uniform(0, 1)

        # Se r for menor que a taxa de seleção, o melhor fitness é selecionado       
        if r <= self.taxa_selecao_melhor:
            pai2 = max(pais_2, key=attrgetter('fitness'))
        else:
            pai2 = min(pais_2, key=attrgetter('fitness'))
        
        # Retorna os pais selecionados através do torneio
        return pai1, pai2

    # Efetua seleção dos pais via Roleta
    def seleciona_pai_rank_linear(self):
        populacao_rank_linear = deepcopy(self.populacao)
        fitness_linear = len(populacao_rank_linear)
        for i in range(len(populacao_rank_linear)):
            populacao_rank_linear[i].fitness = fitness_linear
            fitness_linear = fitness_linear - 1

        soma_avaliacao, valor_ajuste = self.calcular_soma_fitness(populacao_rank_linear)

        # Seleciona o PAI 1
        pai1 = -1
        valor_sorteado = random.random() * soma_avaliacao
        soma = 0
        i = 0
        while i < len(populacao_rank_linear) and soma < valor_sorteado:
            soma += populacao_rank_linear[i].fitness + valor_ajuste
            pai1 += 1
            i += 1
        
        # Seleciona o PAI 2
        pai2 = -1
        valor_sorteado = random.random() * soma_avaliacao
        soma = 0
        i = 0
        while i < len(populacao_rank_linear) and soma < valor_sorteado:
            soma += populacao_rank_linear[i].fitness + valor_ajuste
            pai2 += 1
            i += 1

        # Retorna os pais selecionados pela roleta
        return populacao_rank_linear[pai1], populacao_rank_linear[pai2] 

    # Efetua o crossover de 2 pais, gerando 2 filhos
    def efetuar_crossover(self, pai1, pai2):
        # Se for menor que a taxa de crossover, efetua o crossover
        if random.random() < self.taxa_crossover:
            if self.tipo_crossover == TIPO_CROSSOVER.UNICO_PONTO:
                return self.efetuar_crossover_unico_ponto(pai1, pai2)
            elif self.tipo_crossover == TIPO_CROSSOVER.DUPLO_PONTO:
                return self.efetuar_crossover_duplo_ponto(pai1, pai2)
            elif self.tipo_crossover == TIPO_CROSSOVER.UNIFORME:
                return self.efetuar_crossover_uniforme(pai1, pai2)
            elif self.tipo_crossover == TIPO_CROSSOVER.ARITMETICO:
                return ""
        else: # se não, retorna o próprio pai1 e pai2 (clone)
            return pai1, pai2

        
    # Crossover de Unico Ponto
    def efetuar_crossover_unico_ponto(self, pai1, pai2):
        if (self.crossover_por_entrada): # Crossover de X com X (0 a 10) e de Y com Y (10 a 20)
            # Calcula um local aleatório de corte
            total_bits_x = self.qtde_bits
            local_corte = round(random.random()  * total_bits_x)
            
            # Filho 1
            # X do filho 1
            filho1 = Individuo(self.bits_altura, self.bits_largura, self.bits_folhas, self.numero_geracao)
            x1 = []
            x1 = np.append(pai1.x[0:local_corte] , pai2.x[local_corte:: ])
            filho1.x = x1
            # Y do filho 1
            y1 = []
            y1 = np.append(pai1.y[0:local_corte] , pai2.y[local_corte:: ])
            filho1.y = y1
            filho1.cromossomo = np.append(x1, y1)

            # Filho 2
            # X do filho 2
            filho2 = Individuo(self.bits_altura, self.bits_largura, self.bits_folhas, self.numero_geracao)
            x2 = []
            x2 = np.append(pai2.x[0:local_corte] , pai1.x[local_corte:: ])
            filho2.x = x2
            # Y do filho 2
            y2 = []
            y2 = np.append(pai2.y[0:local_corte] , pai1.y[local_corte:: ])
            filho2.y = y2
            filho2.cromossomo = np.append(x2, y2)

        else: # Crossover em X e Y como um cromossomo s'o
            # Calcula um local aleatório de corte
            total_bits = self.qtde_bits

            local_corte = round(random.random()  * total_bits)

            filho1 = Individuo(self.bits_altura, self.bits_largura, self.bits_folhas, self.numero_geracao)
            xy1 = []
            xy1 = np.append(pai1.cromossomo[0:local_corte] , pai2.cromossomo[local_corte:: ])
            filho1.cromossomo = xy1
            filho1.atualizar()

            filho2 = Individuo(self.bits_altura, self.bits_largura, self.bits_folhas, self.numero_geracao)
            xy2 = []
            xy2 = np.append(pai2.cromossomo[0:local_corte] , pai1.cromossomo[local_corte:: ])
            filho2.cromossomo = xy2
            filho2.atualizar()

        # xa = self.calcular_valor_inteiro(filho1.getProfundidade()) + 1 # altura deve ser no minimo 1
        # ya = self.calcular_valor_inteiro(filho1.getRamos()) + 2 # ramo deve ser no minimo 2
        # xb = self.calcular_valor_inteiro(filho2.getProfundidade()) + 1 # altura deve ser no minimo 1
        # yb = self.calcular_valor_inteiro(filho2.getRamos()) + 2 # ramo deve ser no minimo 2

        # if(xa > 2 or xb > 2 or ya > 5 or yb > 5):
        #     print("")

        novos_filhos = []
        novos_filhos.append(filho1)
        novos_filhos.append(filho2)

        return novos_filhos

    # Efetua o crossover de duplo ponto
    def efetuar_crossover_duplo_ponto(self, pai1, pai2):
        
        if (self.crossover_por_entrada): # Crossover de X com X (0 a 10) e de Y com Y (10 a 20)
            # Calcula um local aleatório de corte
            total_bits_x = self.qtde_bits
            local_corte_1 = round(random.random()  * total_bits_x)
            local_corte_2 = round(random.random()  * total_bits_x)
            
            if (local_corte_1 > local_corte_2):
                corte_ini = local_corte_2
                corte_fim = local_corte_1
            else:
                corte_ini = local_corte_1
                corte_fim = local_corte_2

            # Filho 1
            # X do filho 1
            filho1 = Individuo(self.bits_altura, self.bits_largura, self.bits_folhas, self.numero_geracao)
            x1 = []
            x1 = np.append(pai1.x[0:corte_ini] , pai2.x[corte_ini:corte_fim])
            x1 = np.append(x1, pai1.x[corte_fim::])
            filho1.x = x1
            # Y do filho 1
            y1 = []
            y1 = np.append(pai1.y[0:corte_ini] , pai2.y[corte_ini:corte_fim])
            y1 = np.append(y1, pai1.y[corte_fim::])
            filho1.y = y1
            filho1.cromossomo = np.append(x1, y1)

            # Filho 2
            # X do filho 2
            filho2 = Individuo(self.bits_altura, self.bits_largura, self.bits_folhas, self.numero_geracao)
            x2 = []
            x2 = np.append(pai2.x[0:corte_ini] , pai1.x[corte_ini:corte_fim])
            x2 = np.append(x2, pai2.x[corte_fim::])
            filho2.x = x2
            # Y do filho 2
            y2 = []
            y2 = np.append(pai2.y[0:corte_ini] , pai1.y[corte_ini:corte_fim])
            y2 = np.append(y2, pai2.y[corte_fim::])
            filho2.y = y2
            filho2.cromossomo = np.append(x2, y2)

        else: # Crossover em X e Y como um cromossomo s'o
            # Calcula um local aleatório de corte
            total_bits = self.qtde_bits

            rdn_1 = random.random()
            local_corte_1 = round(rdn_1  * (total_bits-1))
            rdn_2 = random.random()
            local_corte_2 = round(rdn_2  * (total_bits-1))
            
            if (local_corte_1 > local_corte_2):
                corte_ini = local_corte_2
                corte_fim = local_corte_1
            else:
                corte_ini = local_corte_1
                corte_fim = local_corte_2
            
            # filho1 = Individuo(self.bits_altura, self.bits_largura, self.bits_folhas, self.numero_geracao)
            filho1 = Individuo(self.bits_altura, self.bits_largura, self.bits_folhas, self.numero_geracao)
            xy1 = []
            xy1 = np.append(pai1.cromossomo[0:corte_ini] , pai2.cromossomo[corte_ini:corte_fim ])
            xy1 = np.append(xy1, pai1.cromossomo[corte_fim::])
            filho1.cromossomo = xy1
            filho1.atualizar()

            filho2 = Individuo(self.bits_altura, self.bits_largura, self.bits_folhas, self.numero_geracao)
            xy2 = []
            xy2 = np.append(pai2.cromossomo[0:corte_ini] , pai1.cromossomo[corte_ini:corte_fim])
            xy2 = np.append(xy2, pai2.cromossomo[corte_fim::])
            filho2.cromossomo = xy2
            filho2.atualizar()

        novos_filhos = []
        novos_filhos.append(filho1)
        novos_filhos.append(filho2)

        return novos_filhos

    # Efetua o crossover de duplo ponto
    def efetuar_crossover_uniforme(self, pai1, pai2):
        
        filho1 = Individuo(self.bits_altura, self.bits_largura, self.bits_folhas, self.numero_geracao)
        filho2 = Individuo(self.bits_altura, self.bits_largura, self.bits_folhas, self.numero_geracao)

        for i in range(len(pai1.cromossomo)):
            if random.random() < 0.5:
                filho1.cromossomo[i] = pai1.cromossomo[i]
            else:
                filho1.cromossomo[i] = pai2.cromossomo[i]
            
            if random.random() < 0.5:
                filho2.cromossomo[i] = pai1.cromossomo[i]
            else:
                filho2.cromossomo[i] = pai2.cromossomo[i]


        novos_filhos = []
        novos_filhos.append(filho1)
        novos_filhos.append(filho2)

        return novos_filhos


    # baseado em uma taxa de mutação, verifica se um numero randomindo gerado é menor q a taxa. Se for, efetua a mutação. Se não, não.
    def efetuar_mutacao(self, novos_filhos):
        if self.tipo_mutacao == TIPO_MUTACAO.TODOS_GENES: # verifica em cada gene e, se a taxa for ok, ajusta o gene
            for filho in novos_filhos:
                for j in range(len(filho.cromossomo)): # iterage entre os 20 bits
                    # Verifica se efetua a mudança no gene
                    if random.random() < self.taxa_mutacao:
                        if filho.cromossomo[j] == 1:
                            filho.cromossomo[j] = 0
                        else:
                            filho.cromossomo[j] = 1

        elif self.tipo_mutacao == TIPO_MUTACAO.GENE_UNICO: # ajusta um gene
            for filho in novos_filhos:
                if random.random() < self.taxa_mutacao:
                    i_gene = round(random.random() * len(filho.cromossomo)-1)
                    if filho.cromossomo[i_gene] == 1:
                        filho.cromossomo[i_gene] = 0
                    else:
                        filho.cromossomo[i_gene] = 1

        elif self.tipo_mutacao == TIPO_MUTACAO.N_GENES: # Ajusta n genes
            for filho in novos_filhos:
                if random.random() < self.taxa_mutacao:
                    qtd_gene  = round(random.random() * len(filho.cromossomo)-1)
                    for i in range(qtd_gene):
                        i_gene = round(random.random() * len(filho.cromossomo)-1)
                        if filho.cromossomo[i_gene] == 1:
                            filho.cromossomo[i_gene] = 0
                        else:
                            filho.cromossomo[i_gene] = 1


        return novos_filhos

    # Execução do algoritmo genético
    def executar_algoritmo_genetico(self):
        inicio = time.time()
        # Gera a poupalação inicial, de forma aleatória
        self.gerar_populacao_inicial()
        self.avaliar_populacao()
        self.ordenar_populacao()
        self.imprime_melhor_pior_da_geracao()

        for i in range(self.geracoes):
            # soma_fitness = self.soma_total_fitness()
            
            nova_geracao = []
            self.numero_geracao += 1 # Acrescenta o numero da geração

            # Se algum elitismo estiver preparado, guarda os n's melhores
            if(self.qtde_individuo_elitismo > 0):
                for i in range(self.qtde_individuo_elitismo):
                    self.populacao[i].adicionar_geracao()
                    nova_geracao.append(self.populacao[i])

            # Se o Elitismo for 0, gera desde o inicio toda a geração
            # Se alguma qtde de elitismo foi definida, inicia o contador abaixo a partir deste ponto, para manter o tamanho definido da população
            for j in range(self.qtde_individuo_elitismo, self.individuos_por_geracao, 2): # pula de 2 em 2 pois a cada iteração gera 2 filhos
                pai1, pai2 = self.selecionar_pais()

                novos_filhos = self.efetuar_crossover(pai1, pai2)
                novos_filhos = self.efetuar_mutacao(novos_filhos)
                nova_geracao.append(novos_filhos[0])
                nova_geracao.append(novos_filhos[1])

            self.populacao = list(nova_geracao)
            self.avaliar_populacao()
            self.ordenar_populacao()
            self.imprime_melhor_pior_da_geracao()


        fim = time.time()

        # Imprime melhor individuo
        print( "\n" + self.retornar_descricao_tipo_selecao() + " | " + self.retornar_descricao_tipo_crossover())
        print("\nMELHOR INDIVIDUO: \nGeração:%s -> \nFitness: %s \nAltura: %s \nRamos: %s \nValor Altura: %s \nValor Ramos: %s" % (self.melhor_individuo.geracao,
                                                               self.melhor_individuo.fitness,
                                                               self.melhor_individuo.cromossomo[0:self.bits_altura],
                                                               self.melhor_individuo.cromossomo[self.bits_altura:self.bits_altura+self.bits_largura],
                                                               self.calcular_valor_inteiro(self.melhor_individuo.cromossomo[0:self.bits_altura], True),
                                                               self.calcular_valor_inteiro(self.melhor_individuo.cromossomo[self.bits_altura:self.bits_altura+self.bits_largura])))
        self.gerar_grafico(fim-inicio)
        self.salvar_objetos_em_arquivo()

        
        
    def salvar_objetos_em_arquivo(self):
        
        self.salvar_arquivo(self.fitnesses, 'fitnesses.txt')
        self.salvar_arquivo(self.media_fitnesses, 'media_fitnesses.txt')
        self.salvar_arquivo(self.melhores_individuos, 'melhores_individuos.txt')
        self.salvar_arquivo(self.piores_individuos, 'piores_individuos.txt')

    def salvar_arquivo(self, objeto, nome_arquivo ):
        import pickle 
        with open(nome_arquivo, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(objeto, output, pickle.HIGHEST_PROTOCOL)
    
    # carregar - https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence
    # http://www.albertgao.xyz/2016/12/02/how-to-save-a-python-object-to-disk-and-retrieve-it-later/
    # with open('tech_companies.pkl', 'rb') as input:
    #     tech_companies = pickle.load(input)

    def gerar_grafico(self, tempo):
        import plotly
        import plotly.graph_objs as go

        tamanho = len(self.fitnesses)

        trace_melhor = go.Scatter(
            x = np.array(list(range(tamanho))),
            y = np.array(self.fitnesses),
            name= "Melhor",
            mode= "lines+markers"
        )

        trace_media = go.Scatter(
            x = np.array(list(range(tamanho))),
            y = np.array(self.media_fitnesses),
            name= "Media",
            mode= "lines+markers"
        )

        dados_normalizacao  = [trace_melhor, trace_media]


        plotly.offline.plot({
            "data": dados_normalizacao,
            "layout": go.Layout(
                title="Fitness -> <i>Tempo(seg)</i>:" + str(round(tempo,2)) + "<br><b>Individuos por geração</b>: " + str(self.individuos_por_geracao) + "| <b>Tipo Selecao</b>: " + self.retornar_descricao_tipo_selecao() + " | <b>Elitismo</b>: " + str(self.qtde_individuo_elitismo) + "<br><b>Tipo Crossover</b>: " + self.retornar_descricao_tipo_crossover() + " | <b>Tipo Mutação</b>: " + self.retornar_descricao_tipo_mutacao()  + " | <b>Mutacao</b>: " + str(self.taxa_mutacao) ,
                xaxis=dict(
                    title="Geração",
                    showticklabels=True,
                    dtick=1
                ),
                yaxis=dict(
                    title="Fitness" # ,range=[-50, 0]
                )
            )
        }, auto_open=True, filename= "genetico_" + self.retornar_descricao_tipo_selecao()  + "_" + self.retornar_descricao_tipo_crossover() + "_" + str(self.geracoes) + "_" + str(self.individuos_por_geracao)  + ".html")
    

    def retornar_descricao_tipo_selecao(self):
        if self.tipo_selecao == TIPO_SELECAO.ROLETA:
            return "Roleta"
        elif self.tipo_selecao == TIPO_SELECAO.AMOSTRAGEM_ESTOCASTICA:
            return "Amostragem Estocastica"
        elif self.tipo_selecao == TIPO_SELECAO.TORNEIO:
            return "Torneio"
        elif self.tipo_selecao == TIPO_SELECAO.RANK_LINEAR:
            return "Rank Linear"

    def retornar_descricao_tipo_crossover(self):
        if self.tipo_crossover == TIPO_CROSSOVER.UNICO_PONTO:
            return "Unico Ponto"
        elif self.tipo_crossover == TIPO_CROSSOVER.DUPLO_PONTO:
            return "Duplo Ponto"
        elif self.tipo_crossover == TIPO_CROSSOVER.UNIFORME:
            return "Uniforme"
        elif self.tipo_crossover == TIPO_CROSSOVER.ARITMETICO:
            return "Aritmetico"

    def retornar_descricao_tipo_mutacao(self):
        if self.tipo_mutacao == TIPO_MUTACAO.GENE_UNICO:
            return "Testa 1 Gene"
        elif self.tipo_mutacao == TIPO_MUTACAO.TODOS_GENES:
            return "Testa Todos Genes"
        elif self.tipo_mutacao == TIPO_MUTACAO.N_GENES:
            return "Testa N Genes"           


# i1 = Individuo(1,1)
# i1.setXY([0,0])
# i2 = Individuo(1,1)
# i2.setXY([0,10])
# i3 = Individuo(1,1)
# i3.setXY([1,1])
# i4 = Individuo(1,1)
# i4.setXY([0,0])

a = AlgoritmoGenetico(  bits_altura=2, #2
                        bits_largura=1, #3
                        bits_folha_unica=2,
                        geracoes=20, #20,
                        individuos_por_geracao=20, # 30,
                        qtde_bits_por_entrada=2, # 2 bits altura / 3 bits largura 
                        tipo_selecao=TIPO_SELECAO.TORNEIO, # Qual o tipo de seleção aplicada
                        qtde_individuo_elitismo=2,
                        crossover_por_entrada=False, # True ou False. Indica se o crossover será feito por entrada (ex.: POR ENTRADA => pai1.x com pai2.x e pai1.y com pai2.y) ou no cromossomo inteiro (ex.: INTEIRO => pai1.xy com pai2.xy)
                        tipo_crossover=TIPO_CROSSOVER.UNICO_PONTO,
                        taxa_crossover=0.8,
                        taxa_selecao_melhor_torneio=0.8, 
                        tipo_mutacao=TIPO_MUTACAO.TODOS_GENES,
                        taxa_mutacao=0.01)

a.executar_algoritmo_genetico()