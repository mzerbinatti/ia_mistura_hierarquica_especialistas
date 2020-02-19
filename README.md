# Mistura Hierárquica de Especialistas

  Desenvolvimento de algoritmo genético em Python que integra com código Octave (contendo alguns ajustes)

  Este trabalho abordou a utilização do Algoritmo Genético para a identificação de uma boa arquitetura de Mistura Hierárquica de Especialistas, considerando 2 séries que foram utilizadas nos experimentos abaixo. 
  O foco principal que dei foi na geração e criação das possíveis arquiteturas que um MHE poderia ter e, que ao longo da execução do Algoritmo Genético, as arquiteturas de MHE obtivessem melhor Fitness, que neste trabalho utilizei a maximização do likelihood (lli), pudessem se manter e propagar seus genes com maior probabilidade para as seguintes gerações.

  Para este trabalho foi desenvolvido um algoritmo genético em Python e implementadas algumas codificações de cromossomos para que fosse consumido o código de Mistura Hierárquica de Especialistas disponibilizado em MatLab/Octave.  
  Inicialmente, estruturei este trabalho em 3 principais experimentos e estratégias de geração e evolução da arquitetura da MHE: 
- Variação somente da Profundidade e Ramificação da arquitetura da MHE
- Variação da Profundidade, da Ramificação dos Gatings da MHE, e da Ramificação dos Especialistas (folhas) de cada Gating, de forma independente
- Variação total da arquitetura da MHE 
 
