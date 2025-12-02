**PageRank Algorithm**

O PageRank é um algoritmo baseado em grafos desenvolvido por Larry Page e Sergey Brin para medir a importância relativa de nós em uma rede. Ele modela a navegação como um “surfista aleatório” que, a cada passo, segue um link de saída da página atual com certa probabilidade e, com a probabilidade complementar, teleporta para qualquer outro nó do grafo. Dessa forma, páginas (ou vértices) que recebem muitos links de outros nós importantes acumulam um valor de PageRank mais elevado, refletindo maior relevância estrutural dentro da rede.

Matematicamente, o PageRank é definido como a distribuição estacionária de uma cadeia de Markov sobre o grafo, controlada por um fator de amortecimento que equilibra a influência da estrutura de links com o comportamento aleatório do usuário. Essa abordagem reduz o impacto de ruídos locais, é relativamente robusta a manipulações simples de links e permite ranquear nós em redes grandes de forma eficiente, tornando-se um dos algoritmos fundamentais em análise de redes complexas.

**High Energy Physics Citation Network**

Neste projeto, o PageRank é aplicado a uma rede de citações científicas, em que cada nó representa um artigo e cada aresta dirigida A → B indica que o artigo A cita o artigo B. Diferentemente de um problema clássico de classificação supervisionada, o objetivo aqui é medir a importância estrutural dos trabalhos dentro do campo, identificando quais papers funcionam como referências centrais porque são citados por muitos outros artigos relevantes.

A partir do dataset de Física de Altas Energias, a rede de citações é carregada como um grafo dirigido e o algoritmo PageRank é implementado do zero, seguindo o modelo do surfista aleatório. Em seguida, os resultados são comparados com a implementação pronta do NetworkX, analisando-se os dez artigos com maior score e discutindo como a variação do fator de amortecimento afeta o ranqueamento. Dessa forma, o exercício conecta teoria de grafos, métodos numéricos e interpretação substantiva dos nós mais influentes na literatura científica.

Construção do Grafo

O primeiro passo consiste em carregar o arquivo de arestas Cit-HepTh.txt, em que cada linha representa uma citação entre dois artigos. Esse arquivo é lido com o NetworkX e transformado em um grafo dirigido, preservando o sentido das arestas: uma aresta de u para v indica que o paper u cita o paper v.

=== "Code"

``` python
--8<-- "./docs/pagerank/load_graph.py"
```


!!! info
    Essa etapa garante que a estrutura de citações do dataset seja representada corretamente como um grafo dirigido, permitindo aplicar o PageRank diretamente sobre a rede científica.

**Implementação do PageRank From Scratch**

A implementação do PageRank segue a formulação iterativa do modelo do surfista aleatório. Inicialmente, todos os nós recebem o mesmo valor de PageRank, e a cada iteração o score de cada vértice é atualizado a partir das contribuições dos nós que apontam para ele, normalizadas pelo número de saídas. Nós sem arestas de saída (dangling nodes) têm sua massa redistribuída uniformemente por todo o grafo. O processo continua até que a diferença entre duas iterações consecutivas seja menor que uma tolerância pré-definida.

A função abaixo recebe um grafo dirigido do NetworkX e devolve um dicionário com os valores de PageRank para cada nó, permitindo ajustar o fator de amortecimento, a tolerância de convergência e o número máximo de iterações.

=== "Code"

``` python
--8<-- "./docs/pagerank/pagerank_custom.py"
```


!!! info
    Essa implementação reproduz o comportamento do PageRank clássico, incluindo o tratamento de nós sem saída e o uso do damping factor como parâmetro, o que permite posteriormente analisar o impacto de diferentes valores de d na distribuição final de importância.

**Execução dos Experimentos**

Para organizar os experimentos foi criado um script principal responsável por:

Carregar o dataset Cit-HepTh.txt como grafo dirigido.

Aplicar o pagerank_custom para diferentes valores de damping factor (por exemplo, 0.5, 0.85 e 0.99).

Calcular o PageRank usando a função networkx.pagerank com os mesmos parâmetros.

Comparar os resultados das duas abordagens, medindo a diferença máxima entre os vetores de PageRank.

Identificar os dez nós com maior PageRank em cada configuração, permitindo analisar quais artigos se destacam como mais influentes na rede de citações.

=== "Code"

``` python
--8<-- "./docs/pagerank/run_pagerank.py"
```


!!! info
    Esse script centraliza a lógica experimental: leitura do grafo, chamada da implementação própria, comparação com a biblioteca e exibição dos nós mais importantes para cada valor de d.

**Resultados e Análise**

Ao executar o experimento, observa-se que a implementação própria do PageRank converge em poucas iterações e produz valores extremamente próximos aos obtidos pela função networkx.pagerank, com diferenças numéricas muito pequenas. Isso indica que o algoritmo implementado segue corretamente a formulação do PageRank e serve como uma validação prática do método desenvolvido do zero.

A análise dos dez nós com maior PageRank revela os artigos mais centrais na rede de Física de Altas Energias. Esses papers tendem a ser citados por muitos outros trabalhos que também ocupam posições de destaque, formando um núcleo de referências fundamentais na área. Em termos de interpretação, são artigos que funcionam como “hubs” de citação: resultados teóricos importantes, revisões amplamente utilizadas ou contribuições que servem de base para diversos estudos subsequentes.

A variação do damping factor permite observar como o comportamento do surfista aleatório afeta o ranqueamento. Com valores mais baixos, como d = 0.5, a distribuição de PageRank se torna mais uniforme, pois o teleporte ocorre com maior frequência e reduz a influência de caminhos longos de citação. Já com d próximo de 1, como 0.99, o algoritmo privilegia ainda mais a estrutura do grafo, concentrando a importância em componentes fortemente conectadas e aumentando a diferença entre os artigos mais citados e o restante da rede. O valor intermediário d = 0.85, clássico na literatura, oferece um equilíbrio entre exploração aleatória e respeito à estrutura de links, produzindo rankings estáveis e interpretáveis.

De forma geral, o exercício mostra como o PageRank pode ser utilizado além da web, servindo como ferramenta para identificar nós centrais em diferentes tipos de redes. No contexto de citações científicas, ele destaca papers que desempenham papel estruturante na literatura, auxiliando na descoberta de trabalhos seminais e na compreensão da organização do conhecimento em uma área específica.
