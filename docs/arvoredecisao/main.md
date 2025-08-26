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

## Divisão dos Dados

## Treinamento do Modelo

## Avaliação do Modelo

## Conclusão