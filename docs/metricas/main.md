**Metrics and Evaluation - Algoritmo KNN e K-Means**

Os algoritmos K-Nearest Neighbors (KNN) e K-Means são duas abordagens clássicas de Machine Learning que, embora diferentes em sua natureza, compartilham a ideia central de medir a proximidade entre pontos. O KNN é um método supervisionado, utilizado para tarefas de classificação, em que novas amostras são atribuídas à classe predominante entre seus vizinhos mais próximos. Já o K-Means é um método não supervisionado, aplicado para agrupar dados em clusters com base na semelhança de suas características, sem depender de rótulos previamente definidos.

Ao serem aplicados em conjunto, esses algoritmos permitem duas perspectivas complementares: de um lado, a predição direta de categorias por meio do KNN; de outro, a descoberta de padrões ocultos com o K-Means. Essa combinação ajuda tanto a entender melhor a estrutura dos dados quanto a explorar possibilidades de segmentação e análise de comportamento.

**Cars Purchase Decision**

Este projeto tem como objetivo aplicar técnicas de Machine Learning para compreender os fatores que influenciam a decisão de compra de automóveis. A partir de um conjunto de dados com informações sobre idade, gênero e salário anual dos clientes, foi construída uma árvore de decisão capaz de classificar se um indivíduo provavelmente realizará a compra ou não.

## **Exploração dos Dados**

**Estatísticas Descritivas**

Para o projeto foi utilizado o dataset [Cars - Purchase Decision Dataset](https://www.kaggle.com/datasets/gabrielsantello/cars-purchase-decision-dataset){:target='blank'} e contém detalhes de clientes que consideraram comprar um automóvel, juntamente com seus salários.

O conjunto de dados contém **1000 registros** e **5 variáveis**. A variável alvo é **Purchased** (0 = não comprou, 1 = comprou).
Entre as variáveis explicativas, temos Gender (categórica), Age (numérica) e AnnualSalary (numérica).

**Variáveis**

- **User ID:** Código do Cliente

- **Gender:** Gênero do Cliente

- **Age:** Idade do Cliente em anos

- **AnnualSalary:** Salário anual do Cliente

- **Purchased:** Se o cliente realizou a compra

**Estatísticas Descritivas e Visualizações**

O gráfico mostra a relação entre idade e salário dos clientes, destacando quem realizou a compra e quem não comprou:

=== "Result"

    ``` python exec="on" html="1"
    --8<-- "./docs/arvoredecisao/grafdispersao.py"
    ```
=== "Code"

    ``` python
    --8<-- "./docs/arvoredecisao/grafdispersao.py"
    ```

!!! info
    A visualização deixa claro que idade e salário exercem influência relevante no comportamento de compra

O próximo gráfico apresenta a distribuição de clientes por gênero:

=== "Result"

    ``` python exec="on" html="1"
    --8<-- "./docs/arvoredecisao/barras.py"
    ```
=== "Code"

    ``` python
    --8<-- "./docs/arvoredecisao/barras.py"
    ```

!!! info 
    Observa-se que há uma leve predominância de mulheres no dataset.

O último gráfico apresenta a distribuição do salário anual dos clientes, permitindo visualizar a mediana, a dispersão dos valores e a presença de possíveis extremos:

=== "Result"

    ``` python exec="on" html="1"
    --8<-- "./docs/arvoredecisao/boxplot.py"
    ```
=== "Code"

    ``` python
    --8<-- "./docs/arvoredecisao/boxplot.py"
    ```

!!! info
    O gráfico evidencia que a maior parte dos salários está concentrada em uma faixa intermediária, entre aproximadamente 50 mil e 90 mil, com a mediana em torno de 70 mil.

## **Pré-processamento**

Pré-processamento de dados brutos deve ser a primeira etapa ao lidar com datasets de todos tamanhos.

**Data Cleaning**

O processo de data cleaning garante que o conjunto utilizado seja confiável e esteja livre de falhas que possam distorcer os resultados. Consiste em identificar e corrigir problemas como valores ausentes, dados inconsistentes ou informações que não fazem sentido. Essa limpeza permite que a base seja mais fiel à realidade e forneça condições adequadas para a construção de modelos de Machine Learning.

No código, a limpeza foi feita dessa forma: possíveis valores vazios em idade, gênero e salário foram preenchidos com informações representativas, como a mediana ou o valor mais frequente.

=== "Result"

    ``` python exec="on" html="0"
    --8<-- "./docs/arvoredecisao/dataclean.py"
    ```
=== "Code"

    ``` python
    --8<-- "./docs/arvoredecisao/dataclean.py"
    ```

**Encoding Categorical Variables**

O processo de encoding de variáveis categóricas transforma informações em formato de texto em valores numéricos, permitindo que algoritmos de Machine Learning consigam utilizá-las em seus cálculos.

No código, o encoding foi aplicado à variável gênero, convertendo as categorias “Male” e “Female” em valores numéricos (1 e 0). Dessa forma, a base de dados mantém todas as colunas originais, mas agora com a variável categórica representada de maneira adequada para ser usada em algoritmos de classificação.

=== "Result"

    ``` python exec="on" html="0"
    --8<-- "./docs/arvoredecisao/encodcatva.py"
    ```
=== "Code"

    ``` python
    --8<-- "./docs/arvoredecisao/encodcatva.py"
    ```

**Normalização**

A normalização é o processo de reescalar os valores numéricos de forma que fiquem dentro de um intervalo fixo, normalmente entre 0 e 1. Isso facilita a comparação entre variáveis que possuem unidades ou magnitudes diferentes, evitando que atributos com valores muito altos dominem a análise.

No código, a normalização foi aplicada às colunas idade e salário anual, transformando seus valores para a faixa de 0 a 1 por meio do método Min-Max Scaling. Dessa forma, ambas as variáveis passam a estar na mesma escala, tornando o conjunto de dados mais consistente e adequado para a modelagem.

=== "Result"

    ``` python exec="on" html="0"
    --8<-- "./docs/arvoredecisao/normalizacao.py"
    ```
=== "Code"

    ``` python
    --8<-- "./docs/arvoredecisao/normalizacao.py"
    ```

## **Divisão dos Dados**

Após o pré-processamento, o conjunto de dados precisa ser separado em duas partes: uma para treinamento e outra para teste. Essa divisão é fundamental para que o modelo de Machine Learning aprenda padrões a partir de um grupo de exemplos e, depois, seja avaliado em dados que ainda não foram vistos. Dessa forma, é possível medir a capacidade de generalização do modelo e evitar que ele apenas memorize os exemplos fornecidos.

No código, os atributos escolhidos como preditores foram **gênero**, **idade** e **salário anual**, enquanto a variável-alvo foi **Purchased**, que indica se o cliente comprou ou não o produto. A divisão foi feita em **70% para treino** e **30% para teste**, garantindo que a proporção de clientes que compraram e não compraram fosse preservada em ambos os subconjuntos.

=== "Result"

    ``` python exec="on" html="0"
    --8<-- "./docs/arvoredecisao/divisao.py"
    ```
=== "Code"

    ``` python
    --8<-- "./docs/arvoredecisao/divisao.py"
    ```

## **Implementação KNN**

### Usando Scikit-Learn

=== "Result"

    ``` python exec="on" html="0"
    --8<-- "./docs/metricas/knnm.py"
    ```
=== "Code"

    ``` python
    --8<-- "./docs/metricas/knnm.py"
    ```

## **Implementação K-Means**

### Usando Scikit-Learn

A análise com o algoritmo K-Means permitiu identificar dois grupos principais no conjunto de dados, definidos a partir da combinação entre idade e salário anual. Cada cor no gráfico representa um cluster, enquanto as estrelas vermelhas marcam os centróides, ou seja, os pontos médios que caracterizam cada grupo. Essa separação evidencia padrões de comportamento entre os indivíduos, como faixas salariais e idades que tendem a se agrupar. Apesar de não utilizar os rótulos originais (como no KNN), o K-Means oferece uma visão exploratória útil para identificar tendências e estruturas ocultas nos dados.

=== "Gráfico"

    ``` python exec="on" html="1"
    --8<-- "./docs/metricas/kmeansm.py"
    ```
=== "Code"

    ``` python
    --8<-- "./docs/metricas/kmeansm.py"
    ```

As métricas de avaliação desempenham um papel essencial na validação de modelos de machine learning, pois permitem medir sua performance de forma objetiva. No caso do KNN, foram utilizadas métricas supervisionadas como acurácia, matriz de confusão, precisão, recall e F1-score, que mostraram um desempenho consistente, com acurácia em torno de 91% e boa distinção entre as classes 0 (não comprou) e 1 (comprou). 

Já no K-Means, por ser um algoritmo não supervisionado, a análise foi realizada a partir da coerência entre os clusters formados e a variável real de compra, o que evidenciou padrões relevantes no comportamento dos dados. Assim, observa-se que a escolha de métricas adequadas, supervisionadas para classificação e comparativas para clustering, é indispensável para interpretar corretamente os resultados obtidos.
