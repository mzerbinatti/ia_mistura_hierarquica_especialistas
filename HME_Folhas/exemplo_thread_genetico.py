# import random 
# from oct2py import octave
# from oct2py import Oct2Py
# from threading import Thread
# from multiprocessing.pool import ThreadPool
# import multiprocessing as mp
# octave.addpath("C:/Users/miche/Desktop/Octave_Entrega_2")

# # Aqui são as threads. No cálculo de cada fitness dos individuos
# # cria cada um uma instacia do octave e um nucleo executa
# def calcular_funcao(altura, largura, i):
#     print("\nInicio da Thread do Individuo " + str(i))
#     octave = Oct2Py() # tem q declarar para usar outra instancia, senão não adianta nada o paralelismo, pois não o usa. Dessa forma, ele usa.
#     lli = octave.hmeTest(altura, largura) # chama o método
#     print("\nConclusão do Individuo " + str(i))

# # chama a avaliação de cada individuo da população, via thread 
# def avaliar_populacao():
#     MAX_THREADS = mp.cpu_count() # pega a qtde de nucleos disponiveis no PC
#     thread_pool = ThreadPool(processes=MAX_THREADS)
#     individuos_por_geracao = 4 # exemplo
#     for i in range(individuos_por_geracao):
#         altura = int(random.randint(1, 2))
#         largura = int(random.randint(2, 3))
#         thread_pool.apply_async(calcular_funcao, args=([altura, largura, i]))
#     thread_pool.close() # conclui a adição das threads no ppol
#     thread_pool.join() # Espera todas as threads adicionadas acima terminarem


# avaliar_populacao()
# print("FIM")


def calcular_q(min_funcao, max_funcao, qtde_bits_por_entrada):
    q = (max_funcao - min_funcao) / (2**qtde_bits_por_entrada - 1)
    ajuste = min_funcao
    return q, ajuste

def calcular_valor_decimal_q( binario, q, ajuste):
    valor = 0
    exp = 0
    for i in reversed(range(len(binario))):
        valor = valor + (binario[i] * 2**exp) * q
        exp = exp + 1
    return valor + ajuste

q, ajuste = calcular_q(1, 3, 2)
valor1 = calcular_valor_decimal_q([0,0], q, ajuste)
valor2 = calcular_valor_decimal_q([0,1], q, ajuste)
valor3 = calcular_valor_decimal_q([1,0], q, ajuste)
valor4 = calcular_valor_decimal_q([1,1], q, ajuste)
print(valor)
