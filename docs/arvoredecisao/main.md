## Exploração dos Dados

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

## Pré-processamento

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

## Divisão dos Dados

Após o pré-processamento, o conjunto de dados precisa ser separado em duas partes: uma para treinamento e outra para teste. Essa divisão é fundamental para que o modelo de Machine Learning aprenda padrões a partir de um grupo de exemplos e, depois, seja avaliado em dados que ainda não foram vistos. Dessa forma, é possível medir a capacidade de generalização do modelo e evitar que ele apenas memorize os exemplos fornecidos.

No código, os atributos escolhidos como preditores foram gênero, idade e salário anual, enquanto a variável-alvo foi Purchased, que indica se o cliente comprou ou não o produto. A divisão foi feita em 70% para treino e 30% para teste, garantindo que a proporção de clientes que compraram e não compraram fosse preservada em ambos os subconjuntos. Essa separação assegura que a avaliação futura seja mais confiável e representativa.

=== "Result"

    ``` python exec="on" html="0"
    --8<-- "./docs/arvoredecisao/divisao.py"
    ```
=== "Code"

    ``` python
    --8<-- "./docs/arvoredecisao/divisao.py"
    ```

## Treinamento do Modelo

Essa é a árvore de decisão feita:

=== "Decision Tree"

    ``` python exec="on" html="1"
    --8<-- "./docs/arvoredecisao/tree.py"
    ```
=== "Dataset"

    ``` python exec="on" html="0"
    --8<-- "./docs/arvoredecisao/dataset.py"
    ```
=== "Code"

    ``` python
    --8<-- "./docs/arvoredecisao/tree.py"
    ```

## Avaliação do Modelo

## Conclusão