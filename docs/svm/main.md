**Support Vector Machine**

Support Vector Machine (SVM) é um algoritmo supervisionado usado principalmente para classificação que busca encontrar o hiperplano que melhor separa as classes, maximizando a margem entre os pontos de fronteira (os support vectors) e essa linha/hiperplano. Intuitivamente, ele não tenta apenas “acertar os rótulos” no treino, mas encontrar uma separação o mais ampla e estável possível, o que tende a gerar melhor generalização em dados novos. Em cenários com sobreposição ou ruído, o SVM usa o conceito de soft margin, controlado pelo parâmetro C, permitindo alguns erros em troca de uma fronteira mais robusta.

Quando os dados não são linearmente separáveis no espaço original, entra o kernel trick: em vez de desenhar uma reta em 2D, o SVM projeta implicitamente os dados em um espaço de dimensão maior, onde a separação passa a ser linear. O kernel RBF, por exemplo, cria fronteiras de decisão curvas que se adaptam a padrões mais complexos. No contexto do problema de compra de carros, o SVM aprende uma fronteira de decisão não linear em função de gênero, idade e salário anual para distinguir entre clientes que tendem a comprar ou não o veículo, usando apenas alguns pontos-chave (support vectors) para definir essa fronteira.

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

## **Implementação**

=== "Result"

    ``` python exec="on" html="1"
    --8<-- "./docs/svm/svm.py"
    ```
=== "Code"

    ``` python
    --8<-- "./docs/svm/svm.py"
    ```

## **Resultados**

Como não havia um dataset específico disponibilizado para o exercício de SVM, foi reutilizado o mesmo conjunto de dados do projeto de decisão de compra de carros (car_data.csv), contendo informações de gênero, idade, salário anual e a variável alvo Purchased (0 = não comprou, 1 = comprou). A ideia foi treinar um SVM com kernel RBF, implementado do zero, para classificar se um cliente tende ou não a realizar a compra a partir desses atributos.

Os resultados, porém, mostram que o modelo praticamente não conseguiu aprender o padrão dos compradores. A acurácia de teste ficou em torno de 0,5967, mas a matriz de confusão [[179 0] [121 0]] indica que o classificador previu todos os exemplos como “não comprou”. Isso significa que ele acertou os 179 clientes que realmente não compraram, mas errou todos os 121 que compraram (recall da classe Purchased = 1 igual a zero). Na prática, o modelo virou um “classificador da classe majoritária”: funciona bem para identificar quem não compra, mas é inútil para encontrar potenciais compradores, que são justamente o foco do problema de negócio. Esse comportamento sugere que a configuração utilizada (hard margin, parâmetros fixos de kernel, ausência de balanceamento entre classes e de tuning de hiperparâmetros) não foi adequada, e que seriam necessárias etapas adicionais de normalização, ajuste de C e γ e talvez técnicas de balanceamento para obter um SVM realmente útil nesse contexto.