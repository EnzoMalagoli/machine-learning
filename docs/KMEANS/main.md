**Algoritmo K-Means**

O algoritmo K-Means é uma técnica de aprendizado não supervisionado usada para agrupar dados em **k clusters** distintos. Ele funciona localizando os centros (centroides) de cada grupo e atribuindo os pontos ao cluster cujo centro esteja mais próximo, geralmente usando a distância euclidiana. O processo é iterativo: os pontos são realocados conforme os centroides são recalculados, até que os grupos fiquem estáveis.


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

## **Implementação K-Means**

O processo de clustering foi conduzido utilizando Idade e Salário Anual como variáveis principais, resultando em clusters bem definidos e centróides que representam os perfis médios de cada grupo.

Os resultados evidenciaram que a renda anual foi um fator decisivo na formação dos clusters, separando clientes em faixas de poder aquisitivo distintas. Essa segmentação pode ser útil para estratégias de marketing, definição de público-alvo e personalização de ofertas.

=== "Result"

    ``` python exec="on" html="1"
    --8<-- "./docs/KMEANS/km.py"
    ```
=== "Code"

    ``` python
    --8<-- "./docs/KMEANS/km.py"
    ```

O gráfico mostra os clusters formados pelo K-Means, com cada cor representando um grupo de clientes e as estrelas vermelhas indicando os centróides. É possível notar que o Salário Anual foi a variável mais determinante na separação dos grupos, criando um cluster associado a clientes com maior poder aquisitivo e outro relacionado a clientes de renda mais baixa. Assim, o modelo demonstra como o K-Means pode ser usado de forma eficaz para segmentação de mercado e análise de perfis de consumidores.

Apesar dos bons resultados, é importante destacar que o K-Means depende fortemente da escolha do número de clusters (K) e pode ser influenciado por outliers.