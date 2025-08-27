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

**Visualizações**

O gráfico mostra a relação entre idade e salário dos clientes, destacando quem realizou a compra e quem não comprou:

=== "Result"

    ``` python exec="on" html="0"
    --8<-- "./docs/arvoredecisao/grafdispersao.py"
    ```
=== "Code"

    ``` python
    --8<-- "./docs/arvoredecisao/grafdispersao.py"
    ```
Fazer uma avaliação dos resultados do gráfico

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

## Treinamento do Modelo

## Avaliação do Modelo

## Conclusão