import numpy as np

resultado = []
def calcular_arvore(matriz, linha):
    if(linha >= len(matriz)): # Condição de parada da recursão. Se for o último level, ele pára.
        resultado.append(0)
        return
    
    soma_linha = np.sum(matriz[linha])
    if soma_linha <= 1: # Se for <= 1, então é especialista, ou seja, não tem mais de 1 filho.
        # Criar EXPERT
        resultado.append(0)
    else:    
        qtd_coluna = len(matriz[0])
        resultado.append(soma_linha)
        #for coluna in range(len(matriz[0])): # Numero de colunas
        for coluna in range(soma_linha): # Numero de colunas
            linha_ref = (linha * qtd_coluna) + coluna + 1
            calcular_arvore(matriz, linha_ref)
    
    return resultado

arvore = np.array( [[1, 0, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 1, 1],
                    [0, 0, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 0, 1],
                    [0, 0, 1],
                    [1, 0, 0],
                    [0, 0, 0]]
                )
# arvore = np.array([ [1, 1, 1], 
#                     [1, 1, 1],
#                     [1, 1, 1],
#                     [1, 1, 1],
#                     [1, 1, 1],
#                     [1, 1, 1],
#                     [1, 1, 1],
#                     [1, 1, 1],
#                     [1, 1, 1],
#                     [1, 1, 1],
#                     [1, 1, 1],
#                     [1, 1, 1],
#                     [1, 1, 1]
#                 ])
res = calcular_arvore(arvore, 0)
print(res)